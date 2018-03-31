import pylidc as pl
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from functools import reduce
import pickle
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from skimage import transform
plt.interactive(False)

import gc
gc.enable()

def getNoduleSize(nodule):
    # take largest dimension over all annotations
    bb = 0
    for ann in nodule:
        bb = max(bb, max(ann.bbox_dimensions()))
    return bb


def calc_mask_size(mask, mm_per_px):
    size_in_px = np.max([np.max(indices) - np.min(indices) + 1 for indices in np.nonzero(mask)])
    return  size_in_px * mm_per_px


def getLargestSliceInBB(nodule):
    sliceArea = np.sum(nodule.get_boolean_mask().astype('float32'), axis=(0, 1))
    return np.max(sliceArea), np.argmax(sliceArea)


def interpolateZfromBBidx(nodule, zIdx):
    if nodule.get_boolean_mask().shape[2] > 1:
        ratio = zIdx / (nodule.get_boolean_mask().shape[2] - 1)
    else:
        assert zIdx==0
        ratio = 0
    z = nodule.bbox()[2][0] + ratio * (nodule.bbox()[2][1]-nodule.bbox()[2][0])
    return z


def getSlice(dicom, z, rescale=False):
    img_zs = np.array([float(img.ImagePositionPatient[-1]) for img in dicom])
    idx = np.argmin(np.abs(img_zs - z)) # index of closest slice to z

    if rescale:
        di_slice = rescale_di_to_hu(dicom[idx])
    else:
        di_slice = dicom[idx].pixel_array

    return di_slice


def getMask(z, annotation, img_zs, scan):
    mask, bb = annotation.get_boolean_mask(True)
    if z < bb[2][0] or z > bb[2][1]:
        return None, None, None
    curr_slices = list(filter(lambda x: x <= bb[2][1] and x >= bb[2][0], img_zs))
    z_length = len(curr_slices)
    if not (z_length == mask.shape[2]):
        # This block handles the case where
        # the contour annotations "skip a slice".
        old_mask = mask.copy()
        # Create the new mask with appropriate z-length.
        mask = np.zeros((old_mask.shape[0],
                         old_mask.shape[1],
                         z_length), dtype=np.bool)
        # Map z's to an integer.
        z_to_index = dict(zip(
            curr_slices,
            range(z_length)
        ))
        contour_zs = np.unique([c.image_z_position for c in annotation.contours])
        # Map each slice to its correct location.
        for k in range(old_mask.shape[2]):
            cz = contour_zs[k]
            if cz in z_to_index.keys():
                index_of_contour = z_to_index[cz]
            else:
                # handles miss-labeled ImagePositionPatient
                fnames = scan.sorted_dicom_file_names.split(',')
                assert (len(img_zs) == len(fnames))
                z_to_contor_index = dict(zip(contour_zs, range(len(contour_zs))))
                index_of_contour = [fnames.index(c.dicom_file_name) for c in annotation.contours]
            mask[:, :, index_of_contour] = old_mask[:, :, k]
        # Get rid of the old one.
        del old_mask
    mask_idx = np.argwhere(z == np.array(curr_slices))
    assert (1 == len(mask_idx))
    slice_areas = [np.count_nonzero(mask[:, :, int(idx)]) for idx in range(z_length)]
    weights = slice_areas / np.max(slice_areas)
    return mask[:, :, int(mask_idx)], bb[0:2], weights[int(mask_idx)]


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


def crop(image, mask, fix_size=None, min_size=0, ratio=1.0, stdev=0.15):
    # Size:
    #   first sampled as Uniform(mask-size, full-patch-size)
    #   than clipped by [min_size] to more weight to the nodule size
    # ratio:
    #   Each dimension size is sampled so as to conform to [1-ratio, 1+ratio] relation
    # the shift of the crop window is sampled with normal distribution to have a bias towards a centered crop
    mask_y = np.where(np.sum(mask, axis=1))[0]
    mask_x = np.where(np.sum(mask, axis=0))[0]
    if fix_size is None:
        # Determine Size boundries on Y axis
        max_size_y = image.shape[0]
        min_size_y = np.max(mask_y) - np.min(mask_y)
        #print('min_size_y, max_size_y={}, {}'.format(min_size_y, max_size_y))
        # Determine Size boundries on X axis
        max_size_x = image.shape[1]
        min_size_x = np.max(mask_x) - np.min(mask_x)
        #print('min_size_x, max_size_x={}, {}'.format(min_size_x, max_size_x))
        # correct for ratio
        min_size_y = np.maximum(min_size_y, (1-ratio)*min_size_x)
        max_size_y = np.minimum(max_size_y, (1+ratio)*max_size_x)
        #print('min_size_y, max_size_y={}, {}'.format(min_size_y, max_size_y))

        # Determine New Size on Y axis
        new_size_y = np.random.rand() * (max_size_y - min_size_y) + min_size_y
        new_size_y = np.maximum(new_size_y, min_size).astype('uint16')
        #print('new_size_y={}'.format(new_size_y))
        # correct for ratio (based on chosen size of Y
        min_size_x = np.maximum(min_size_x, (1 - ratio) * new_size_y)
        max_size_x = np.minimum(max_size_x, (1 + ratio) * new_size_y)
        #print('min_size_x, max_size_x={}, {}'.format(min_size_x, max_size_x))
        # Determine New Size on X axis
        new_size_x = np.random.rand() * (max_size_x - min_size_x) + min_size_x
        new_size_x = np.maximum(new_size_x, min_size).astype('uint16')
        #print('new_size_x={}'.format(new_size_x))
    else:
        new_size_x = fix_size
        new_size_y = fix_size

    # Determine crop translation on Y axis
    min_crop_start_y = np.maximum(0, np.max(mask_y) - new_size_y)
    max_crop_start_y = np.minimum(np.min(mask_y), image.shape[0] - new_size_y)
    rand_factor = np.maximum(0.0, np.minimum(1.0, stdev*np.random.normal()+0.5))
    crop_start_y     = min_crop_start_y + rand_factor * (max_crop_start_y - min_crop_start_y)
    crop_start_y = crop_start_y.astype('uint16')
    #print('min_crop_start_y, max_crop_start_y={}, {}'.format(min_crop_start_y, max_crop_start_y))
    #print('factor={}'.format(rand_factor))
    #print('crop_start_y={}'.format(crop_start_y))

    # Determine crop translation on X axis
    min_crop_start_x = np.maximum(0, np.max(mask_x) - new_size_x)
    max_crop_start_x = np.minimum(np.min(mask_x), image.shape[1] - new_size_x)
    rand_factor = np.maximum(0.0, np.minimum(1.0, stdev * np.random.normal() + 0.5))
    crop_start_x     = min_crop_start_x + rand_factor * (max_crop_start_x - min_crop_start_x)
    crop_start_x = crop_start_x.astype('uint16')
    #print('min_crop_start_x, max_crop_start_x={}, {}'.format(min_crop_start_x, max_crop_start_x))
    #print('factor={}'.format(rand_factor))
    #print('crop_start_x={}'.format(crop_start_x))

    assert (crop_start_x >= 0)
    assert (crop_start_y >= 0)
    assert ( (crop_start_y + new_size_y) <= image.shape[0])
    assert ( (crop_start_x + new_size_x) <= image.shape[1])
    new_image = image[crop_start_y:crop_start_y + new_size_y, crop_start_x:crop_start_x + new_size_x]
    new_mask  =  mask[crop_start_y:crop_start_y + new_size_y, crop_start_x:crop_start_x + new_size_x]

    return new_image, new_mask


def rescale_di_to_hu(diEntry):
    image = diEntry.pixel_array
    intercept, slope = diEntry.RescaleIntercept, diEntry.RescaleSlope

    return rescale_im_to_hu(image, intercept, slope)


def rescale_im_to_hu(image, intercept, slope):
    image[image == -2000] = -1024 - intercept
    image = slope * image.astype(np.float64)
    image = image.astype(np.int16) + np.int16(intercept)

    return image


def get_z_range(annotations):
    aggregated_bb = np.vstack([ann.bbox()[2] for ann in annotations])
    return np.min(aggregated_bb[:, 0]), np.max(aggregated_bb[:, 1])


# ----- Main -----
# ----------------

def extract_from_cluster_map(cluster_map, patch_size=144, res='Legacy', dump=True):

    filename = 'NodulePatchesNew{}-{}.p'.format(patch_size, res)
    dataset = []
    if dump is False:
        print("Running without dump")

    for scan in pl.query(pl.Scan).all()[:]:
    # cycle 1018 scans
    #
    # Example for debuging:
    #   scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == 'LIDC-IDRI-0004').first()
    #
        try:
            nods, cluster_indices = cluster_map[scan.id]
        except:
            continue
        print("Study ({}), Series({}) of patient {}:".format(scan.study_instance_uid, scan.series_instance_uid, scan.patient_id))
        dicom = scan.load_all_dicom_images(verbose=False)

        for indices in cluster_indices:
            assert len(nods) > 0
            nodules_in_cluster = np.concatenate([nods[i] for i in indices])
            print("\tCluster with {} nodules.".format(len(nodules_in_cluster)))

            z_range = get_z_range(nodules_in_cluster)
            img_zs = [float(img.ImagePositionPatient[-1]) for img in dicom]
            assert(len(np.unique(img_zs)) == len(img_zs))
            for z in filter(lambda x: x <= z_range[1] and x >= z_range[0], img_zs):
                image = getSlice(dicom, z, rescale=True)
                full_mask = np.zeros(image.shape).astype('bool')
                weights = []
                ratings = []
                nodule_ids = []
                annotation_size = []
                for nod in nodules_in_cluster:
                    mask, bb, w = getMask(z, nod, img_zs, scan)
                    if mask is None or 0 == w: # skip annotation
                        continue
                    full_mask[int(bb[0][0]):int(bb[0][1]+1), int(bb[1][0]):int(bb[1][1]+1)] |= mask
                    nodule_ids += [nod._nodule_id]
                    ratings += [nod.feature_vals()]
                    assert(len(np.flatnonzero(mask)) > 0)
                    annotation_size += [calc_mask_size(mask, mm_per_px=scan.pixel_spacing)]
                    weights += [w]
                if 0 == np.count_nonzero(full_mask): # skips slice
                    continue
                mask_size = calc_mask_size(full_mask, mm_per_px=scan.pixel_spacing)
                if type(res) is float:
                    new_shape = tuple((np.array(image.shape) * (scan.pixel_spacing / res)).astype('int'))
                    image =     transform.resize(image,     output_shape=new_shape, order=2, preserve_range=True, mode='constant')
                    full_mask = transform.resize(full_mask, output_shape=new_shape, order=0, preserve_range=True, mode='constant')
                    if 0 == np.count_nonzero(full_mask):
                        # sometimes the mask is pixel-wide, so after resize nothing is left
                        # would've anyhow been filtered in later stages
                        continue
                patch, mask = crop(image, full_mask, fix_size=patch_size, stdev=0)
                patch = rescale_im_to_hu(patch, dicom[0].RescaleIntercept, dicom[0].RescaleSlope)
                if np.abs(mask_size - calc_mask_size(mask, mm_per_px=res)) > res:
                    print("{}, {}:\n\tfull mask size = {}\n\tmask size = {}".format(scan.patient_id, z, mask_size, calc_mask_size(mask, mm_per_px=res)))
                assert(patch.shape == (patch_size, patch_size))
                assert(mask.shape == (patch_size, patch_size))

                entry = {
                    'patch':    patch.astype(np.int16),
                    'info':     (scan.patient_id, scan.study_instance_uid, scan.series_instance_uid, nodule_ids),
                    'nod_ids':  nodule_ids,
                    'rating':   np.array(ratings),
                    'ann_size': np.array(annotation_size),
                    'weights':  np.array(weights),
                    'mask':     mask.astype(np.int8),
                    'z':        z,
                    'size':     mask_size
                }
                dataset.append(entry)

    print("Prepared {} entries".format(len(dataset)))

    if dump:
        pickle.dump(dataset, open(filename, 'wb'))
        print("Dumped to {}".format(filename))
    else:
        print("No Dump")


def extract(patch_size=144, res='Legacy', dump=True):

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
    #
    # Example for debuging:
    #   scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == 'LIDC-IDRI-0004').first()
    #
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


def cluster_by_cliques(adjacency, nodes):
    G = nx.from_numpy_matrix(adjacency)
    cliques = list(nx.find_cliques(G))
    return cliques

'''
    seeds = [nodes[0]]
    all_seeds = [nodes[0]]
    clusters = []
    while len(seeds) > 0:
        id = seeds.pop()
        clique = [id]
        for adj_id in nodes:
            if id == adj_id:
                continue
            match = True
            for c in clique:
                if 0 == adjacency[c, adj_id]:
                    match = False
                    break
            if match:
                clique.append(adj_id)
                all_seeds.append(adj_id)
            else:
                if adj_id in all_seeds:
                    continue
                else:
                    seeds.append(adj_id)
                    all_seeds.append(adj_id)
        clusters.append(clique)
    return clusters
'''


def plot2d_mds(clusters, data=None, distance_matrix=None):
    if data is None and distance_matrix is None:
        return
    elif distance_matrix is None:
        distance_matrix = squareform(pdist(data, 'euclidean'))
    mds = MDS(n_components=2, max_iter=500, metric=True, dissimilarity="precomputed")
    proj = mds.fit_transform(distance_matrix)
    plt.scatter(proj[:, 0], proj[:, 1], s=1, c='black')
    size = 100
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'orange']
    for idx, cluster in enumerate(clusters):
        plt.scatter(proj[cluster, 0], proj[cluster, 1], marker="o", s=size, facecolors='none', edgecolors=colors[idx % 6])
        size += 200


def mds(scan, clusters, distance_matrix):
    all_anns = scan.annotations
    f0 = np.min([n.id for n in all_anns])
    data = np.array([all_anns[i-f0].feature_vals() for i in np.array([n.id for n in all_anns])])

    plt.figure()
    plt.title("MDS Projection: {}".format(scan.patient_id))
    plt.subplot(121)
    plt.title("Rating")
    plot2d_mds(clusters, data=data)
    plt.subplot(122)
    plt.title("Distance (Intersection)")
    plot2d_mds(clusters, distance_matrix=distance_matrix)


def remove_duplicates_and_subsets(clusters, n):
    # map indices to binary list
    mapped_ = [[1 if (i - 1) in indices else 0 for i in range(n, 0, -1)] for indices in clusters]
    # map binary to integer (unique hash)
    to_hash = [2 ** (i - 1) for i in range(n, 0, -1)]
    hash = [np.array(m).dot(to_hash) for m in mapped_]
    # remove duplicates
    hash = list(set(hash))
    # remove subsets
    hash_np = np.array(hash)
    clean_hash = hash_np[np.array([np.count_nonzero(h == [h & a for a in hash]) for h in hash]) == 1]
    # map back to indices
    clean_clusters = [list(np.concatenate(len(bin(h)) - 3 - np.argwhere([1 if digit == '1' else 0 for digit in bin(h)[2:]])))
                      for h in clean_hash]
    return clean_clusters


def check_nodule_intersections(patch_size  = 144, res ='Legacy'):
    recluster_using_cliques = False
    pat_with_nod    = 0
    pat_without_nod = 0
    nodule_count    = 0
    max_size        = 0
    min_size        = 999999
    min_dist        = 999999
    outliers = []
    size_list = []
    global_size_list = []

    pause = 0
    for scan in pl.query(pl.Scan).all()[:]:
        if len(scan.annotations) == 0:
            continue
        # cluster by intersection
        tol = 0.95
        nods, D = scan.cluster_annotations(metric='jaccard', tol=tol, return_distance_matrix=True)
        if len(nods) == 0:
            pat_without_nod += 1
            continue
        pat_with_nod += 1

        if recluster_using_cliques:
            adjacency = D <= tol
            if adjacency.shape[0] > 1:
                clusters = cluster_by_cliques(adjacency, None)
                print("Study ({}), Series({}) of patient {}: {} connected components. {} cliques"
                      .format(scan.study_instance_uid, scan.series_instance_uid, scan.patient_id, len(nods), len(clusters)))
                nodule_count += len(nods)
                if len(nods) != len(clusters): #[[n.id for n in anns] for anns in nods]
                    pause = pause + 1
                    mds(scan=scan, clusters=clusters, distance_matrix=D)
                # re-cluster nodules by cliques
                nods = [[scan.annotations[i] for i in ids] for ids in clusters]
            else:
                clusters = [[0]]
        else:
            id_0 = np.min([ann.id for ann in scan.annotations])
            clusters = [[ann.id - id_0 for ann in cluster] for cluster in nods]

        centers = []
        boxes   = []
        for cluster in clusters:
            nod = [scan.annotations[ann_id] for ann_id in cluster]
            print("Nodule of patient {} with {} annotations.".format(scan.patient_id, len(nod)))
            min_ = reduce((lambda x, y: np.minimum(x , y)), [ann.bbox()[:, 0] for ann in nod])
            max_ = reduce((lambda x, y: np.maximum(x, y)),  [ann.bbox()[:, 1] for ann in nod])
            size = scan.pixel_spacing*(max_ - min_ + 1)
            size_list.append(size)
            if np.max(size) >= 64:
                print("\tNodule Size = {:.1f} x {:.1f} x {:.1f}".format(size[0], size[1], size[2]))
            if size[2] == 1:
                print("\t\tNodule BB = {}".format([ann.bbox()[:, 0] for ann in nod]))
            max_size = np.maximum(max_size, size)
            min_size = np.minimum(min_size, size)

            centers.append(scan.pixel_spacing*min_ + size//2)
            boxes.append( np.vstack([scan.pixel_spacing*min_, scan.pixel_spacing*max_]) )

        cluster_candidates = []
        for i, nod_i in enumerate(nods):
            j_outs = []
            for j, nod_j in enumerate(nods):
                if i==j:
                    continue
                #if centers[i][2] < boxes[j][0][2]: # ignore if cross-section of i doesn't contain j
                #    continue
                #if centers[i][2] > boxes[j][1][2]: # ignore if cross-section of i doesn't contain j
                #    continue
                dist = np.abs(centers[i] - boxes[j])
                dist = np.min(dist, axis=0)
                dist = np.max(dist)
                min_dist = np.minimum(min_dist, dist)
                if dist > 32:
                    continue
                if dist > 10:
                    stop = 1
                print("\tDist = {}".format(dist))
                min_ = np.minimum(boxes[i][0], boxes[j][0])
                max_ = np.maximum(boxes[i][1], boxes[j][1])
                size = (max_ - min_ + 1)
                print("\t\tMerged ({}, {}) Size = {:.1f} x {:.1f} x {:.1f}".format(i, j, size[0], size[1], size[2]))
                j_outs.append(j)
                outliers.append((dist, np.max(size)))
            if len(j_outs) > 1:
                boxes = np.array(boxes)
                min_ = reduce((lambda x, y: np.minimum(x, y)), [bb[0, :] for bb in boxes[j_outs+[i]]])
                max_ = reduce((lambda x, y: np.maximum(x, y)), [bb[1, :] for bb in boxes[j_outs+[i]]])
                size = (max_ - min_ + 1)
                if np.any(size > 60):
                    stop = 1
                print("\t\t Global Merged ({}, {}) Size = {:.1f} x {:.1f} x {:.1f}".format(i, j_outs, size[0], size[1], size[2]))
                global_size_list.append(np.max(size))

    print("="*30)
    print("Prepared {} entries".format(nodule_count))
    print("{} patients with nodules, {} patients without nodules".format(pat_with_nod, pat_without_nod))
    print("\tMax Size = {:.1f} x {:.1f} x {:.1f}".format(max_size[0], max_size[1], max_size[2]))
    print("\tMin Size = {:.1f} x {:.1f} x {:.1f}".format(min_size[0], min_size[1], min_size[2]))
    print("\tMin Dist = {}".format(min_dist))
    print("== Number of cluster breaks = {} ==".format(pause))

    x_dist = [o[0] for o in outliers]
    y_size = [o[1] for o in outliers]

    plt.figure()

    plt.subplot(311)
    plt.title('Nodule (cluster) Size')
    plt.xlabel('size')
    plt.ylabel('hist')
    plt.hist(np.max(size_list, axis=1), 50)

    plt.subplot(312)
    plt.title('Pairwise-Merges')
    plt.xlabel('dist')
    plt.ylabel('merged size')
    plt.scatter(np.array(x_dist).astype('uint'), np.array(y_size).astype('uint'))

    plt.subplot(313)
    plt.title('Total Size')
    plt.xlabel('size')
    plt.ylabel('hist')
    plt.hist(global_size_list, 50)

    plt.show()


def cluster_all_annotations(size_mm  = 64):
    annotation_cluster_map = {}
    for scan in pl.query(pl.Scan).all()[:]:
        if len(scan.annotations) == 0:
            continue
        print("Patient {}:".format(scan.patient_id))
        # cluster by intersection
        nods = scan.cluster_annotations(metric='jaccard', tol=0.95, return_distance_matrix=False)
        id_0 = np.min([ann.id for ann in scan.annotations])
        clusters = [[ann.id - id_0 for ann in cluster] for cluster in nods]

        # calculate bb for each cluster
        centers = []
        boxes   = []
        for cluster in clusters:
            nod = [scan.annotations[ann_id] for ann_id in cluster]
            print("\tNodule with {} annotations.".format(len(nod)))
            min_ = reduce((lambda x, y: np.minimum(x , y)), [ann.bbox()[:, 0] for ann in nod])
            max_ = reduce((lambda x, y: np.maximum(x, y)),  [ann.bbox()[:, 1] for ann in nod])
            size = scan.pixel_spacing*(max_ - min_ + 1)
            centers.append(scan.pixel_spacing*min_ + size//2)
            boxes.append( np.vstack([scan.pixel_spacing*min_, scan.pixel_spacing*max_]) )

        # check if there are nearby clusters
        # to create cluster groups
        cluster_candidates = []
        for i, nod_i in enumerate(nods):
            j_outs = []
            for j, nod_j in enumerate(nods):
                if i==j:
                    continue
                if boxes[i][1][2] < boxes[j][0][2]: # ignore if cross-section of i doesn't contain j
                    continue
                if boxes[i][0][2] > boxes[j][1][2]: # ignore if cross-section of i doesn't contain j
                    continue
                dist = np.abs(centers[i] - boxes[j])
                dist = np.min(dist, axis=0)
                dist = np.max(dist)
                if dist > size_mm//2:
                    continue
                j_outs.append(j)
            cluster_candidates.append([i]+j_outs)
        cluster_candidates = remove_duplicates_and_subsets(cluster_candidates, len(nods))
        annotation_cluster_map[scan.id] = (nods, cluster_candidates)
        for cluster in cluster_candidates:
            print("\tCluster: {}".format(cluster))

    return annotation_cluster_map
