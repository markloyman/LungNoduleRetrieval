import pylidc as pl
import numpy as np
import matplotlib.pyplot as plt

#plt.interactive(False)

def getNoduleSize(nodule):
    # take largest dimension over all annotations
    # nodule: cluster of annotations
    bb = 0
    for ann in nodule:
        bb = max(bb, max(ann.bbox_dimensions()))
    return bb

def getAnnotation(info, return_all=False, nodule_ids=None):
    qu = pl.query(pl.Scan).filter(pl.Scan.patient_id == info[0],
                                  pl.Scan.study_instance_uid == info[1],
                                  pl.Scan.series_instance_uid == info[2])
    assert (qu.count() == 1)
    scan = qu.first()

    anns = []

    if return_all is True:
        if nodule_ids is None:
            print('WRN: Returning ALL annotations from scan')
            anns= scan.annotations

        else:
            nodule_id_list = [a._nodule_id for a in scan.annotations]
            for nd_id in nodule_ids:
                if nd_id in nodule_id_list:
                    index = nodule_id_list.index(nd_id)
                    anns.append(scan.annotations[index])
                else:
                    print("Nodule '{}' not found".format(nd_id))

    else:
    # return_all is False
        if info[3] is None:
            print('WRN: Returning ALL annotations from scan')
            anns = scan.annotations
        else:
            nodule_id_list = [ann._nodule_id for ann in scan.annotations]
            if type(info[3]) is str:
                nodule_id = info[3]
            elif type(info[3]) is list:
                nodule_id = info[3][0]
            try:
                id = nodule_id_list.index(nodule_id)
                anns = scan.annotations[id]
            except:
                print("Nodule '{}' not found".format(info[3]))
                anns = None

    #if info[3] is not None:
    #    if return_all:
    #        ann_clusters = scan.cluster_annotations()
    #        for clust in ann_clusters:
    #            if info[3] in [ann._nodule_id for ann in clust]:
    #               return clust
    #        return None
    #    else:
    #        nodule_id_list = [ann._nodule_id for ann in scan.annotations]
    #        if info[3] in nodule_id_list:
    #            id = nodule_id_list.index(info[3])
    #            ann = scan.annotations[id]
    #        else:
    #            return None
    #else:
    #    ann = scan.annotations

    return anns


def check_patch(entry, in_dicom=False):
    info = entry['info']
    print(info)
    print('Patch Stats: Mean={}, Min={}, Max={}'.format( np.mean(entry['patch']),
                                                         np.min(entry['patch']),
                                                         np.max(entry['patch'])))
    plt.figure('Patch (P: {})'.format(info[0]))
    plt.imshow(entry['patch']*(0.4 + 0.6*entry['mask']), cmap='gray')
    plt.title('Patch: {}'.format(np.mean(entry['rating'], axis=0)))

    if in_dicom:
        getAnnotation(info).visualize_in_scan()

    plt.show()


def calc_rating(meta_data, nodule_ids = None, method='mean'):
    if method is not 'single':
        #assert(nodule_ids is not None)
        if nodule_ids is None:
            nodule_ids = meta_data[3]
    if isinstance(nodule_ids, str):
        nodule_ids = [nodule_ids]
    if method is 'single':
        # currently only one of the ratings is taken into account
        ann     = getAnnotation(meta_data, return_all=False)
        rating  = ann.feature_vals()
    elif method is 'malig':
        all_anns = getAnnotation(meta_data, nodule_ids=nodule_ids, return_all=True)
        rating = np.mean([a.feature_vals() for a in all_anns], axis=(0))
        rating = rating[-1]
    elif method is 'mean':
        all_anns = getAnnotation(meta_data, nodule_ids=nodule_ids, return_all=True)
        rating   = np.mean([a.feature_vals() for a in all_anns], axis=(0))
    elif method is 'raw':
        all_anns = getAnnotation(meta_data, nodule_ids=nodule_ids, return_all=True)
        rating   = [a.feature_vals() for a in all_anns]
    elif method is 'confidence':
        all_anns = getAnnotation(meta_data, nodule_ids=nodule_ids, return_all=True)
        rating   = np.std([a.feature_vals() for a in all_anns], axis=(0))
    else:
        print('Illegal method - {}'.format(method))
        assert(False)
        rating = None

    return np.array(rating)


'''
def getScanStats(info_list):
    for info in info_list:
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == info[0],
                                      pl.Scan.study_instance_uid == info[1],
                                      pl.Scan.series_instance_uid == info[2]).first()

        scan.pixel_spacing

'''
