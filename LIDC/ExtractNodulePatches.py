import pylidc as pl
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.interactive(False)

import gc
gc.enable()

def getNoduleSize(nodule):
    # take largest dimension over all annotations
    bb = 0
    for ann in nodule:
        bb = max(bb, max(ann.bbox_dimensions()))
    return bb


def getLargestSliceInBB(nodule):
    sliceArea = np.sum(nodule.get_boolean_mask().astype('float32'), axis=(0, 1))
    return (np.max(sliceArea), np.argmax(sliceArea))


def interpolateZfromBBidx(nodule, zIdx):
    if nodule.get_boolean_mask().shape[2] > 1:
        ratio = zIdx / (nodule.get_boolean_mask().shape[2] - 1)
    else:
        assert zIdx==0
        ratio = 0
    z = nodule.bbox()[2][0] + ratio * (nodule.bbox()[2][1]-nodule.bbox()[2][0])
    return z


def getSlice(dicom, z, rescale=False):
    img_zs = [float(img.ImagePositionPatient[-1]) for img in dicom]
    idx = np.argmin(np.abs(img_zs - z)) # index of closest slice to z

    if rescale:
        di_slice = rescale_di_to_hu(dicom[idx])
    else:
        di_slice = dicom[idx].pixel_array

    return di_slice


def get_full_size_mask(nodule, size):
    bb = nodule.bbox().astype('uint')
    mask = np.zeros(shape=size).astype('uint')
    mask[bb[1][0]:bb[1][1], bb[0][0]:bb[0][1]] = 1
    return mask


def cropSlice(slice, center, size):
    cx, cy = max(center[0], size/2), max(center[1], size/2)
    cx, cy = min(cx, slice.shape[0] - size/2), min(cy, slice.shape[1] - size/2)

    x0 = np.round(cx - size/2).astype('uint')
    y0 = np.round(cy - size/2).astype('uint')

    return slice[y0:y0+size, x0:x0+size]


def rescale_di_to_hu(diEntry):
    image = diEntry.pixel_array
    intercept, slope = diEntry.RescaleIntercept, diEntry.RescaleSlope

    return rescale_im_to_hu(image, intercept, slope)


def rescale_im_to_hu(image, intercept, slope):
    image[image == -2000] = -1024 - intercept
    image = slope * image.astype(np.float64)
    image = image.astype(np.int16) + np.int16(intercept)

    return image

# ----- Main -----
# ----------------


def extract(patch_size  = 144, res ='Legacy', dump = True):

    filename = 'NodulePatches{}-{}.p'.format(patch_size, res)

    dataset = []
    nodSize = []
    pat_with_nod    = 0
    pat_without_nod = 0
    patient_nodules = {}

    if dump is False:
        print("Running without dump")

    for scan in pl.query(pl.Scan).all()[:]:
    # cycle 1018 scans
        nods = scan.cluster_annotations(metric='jaccard', tol=0.95, tol_limit=0.7)
        if len(nods) > 0:
            pat_with_nod += 1
            print("Study ({}), Series({}) of patient {}: {} nodules."
                  .format(scan.study_instance_uid, scan.series_instance_uid, scan.patient_id, len(nods)))
            patient_nodules['scan.patient_id'] = len(nods)
            dicom = scan.load_all_dicom_images(verbose=False)

            for nod in nods:
                print("Nodule of patient {} with {} annotations.".format(scan.patient_id, len(nod)))
                largestSliceA = [getLargestSliceInBB(ann)[0] for ann in nod] # larget slice within annotated bb
                annID = np.argmax(largestSliceA) # which of the annotation has the largest slice

                largestSliceZ = [getLargestSliceInBB(ann)[1] for ann in nod]  # index within the mask
                z   = interpolateZfromBBidx(nod[annID], largestSliceZ[annID]) # just for the entry data
                # possible mismatch betwean retrived z and largestSliceZ[annID] due to missing dicom files
                #
                if res is 'Legacy':
                    di_slice = getSlice(dicom, z, rescale=True)
                    mask  = get_full_size_mask(nod[annID], di_slice.shape)
                    patch = cropSlice(di_slice, nod[annID].centroid(), patch_size)
                    mask = cropSlice(mask, nod[annID].centroid(), patch_size)
                else:
                    vol0, seg0 = nod[annID].uniform_cubic_resample(side_length=(patch_size - 1), resolution=res, verbose=0)
                    largestSliceZ = np.argmax(np.sum(seg0.astype('float32'), axis=(0, 1)))
                    patch = rescale_im_to_hu(vol0[:, :, largestSliceZ], dicom[0].RescaleIntercept, dicom[0].RescaleSlope)
                    mask  = seg0[:, :, largestSliceZ]

                entry = {
                    'patch':    patch.astype(np.int16),
                    'info':     (scan.patient_id, scan.study_instance_uid, scan.series_instance_uid, nod[annID]._nodule_id),
                    'nod_ids':  [n._nodule_id for n in nod],
                    'rating':   np.array([ann.feature_vals() for ann in nod]),
                    'mask':     mask.astype(np.int16),
                    'z':        z,
                    'size':     getNoduleSize(nod)
                }
                dataset.append(entry)

                #gc.collect()
        else:
            pat_without_nod += 1

    print("Prepared {} entries".format(len(dataset)))
    print("{} patients with nodules, {} patients without nodules".format(pat_with_nod, pat_without_nod))

    if dump:
        pickle.dump(dataset, open(filename, 'wb'))
        print("Dumpted to {}".format(filename))
    else:
        print("No Dump")