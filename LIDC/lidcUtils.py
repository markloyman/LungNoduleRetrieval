import pylidc as pl
import numpy as np
import matplotlib.pyplot as plt

#plt.interactive(False)

def getAnnotation(info, return_all=False):
    qu = pl.query(pl.Scan).filter(pl.Scan.patient_id == info[0],
                                  pl.Scan.study_instance_uid == info[1],
                                  pl.Scan.series_instance_uid == info[2])
    assert (qu.count() == 1)
    scan = qu.first()

    if (info[3] is not None) and (return_all is False):
        nodule_id_list =  [ann._nodule_id for ann in scan.annotations]
        if info[3] in nodule_id_list:
            id =  nodule_id_list.index(info[3])
            ann = scan.annotations[id]
        else:
            return None
    else:
        ann = scan.annotations

    return ann

def CheckPatch(entry):
    info = entry['info']
    print(info)
    ann = getAnnotation(info)
    ann.visualize_in_scan()
    plt.figure('Patch')
    plt.imshow(entry['patch']+1000*entry['mask'])
    plt.title('Patch')
    plt.show()


def calc_rating(meta_data, method='mean'):
    if method is 'single':
        # currently only one of the ratings is taken into account
        ann = getAnnotation(meta_data, return_all=False)
        rating = ann.feature_vals()
    elif method is 'malig':
        ann = getAnnotation(meta_data)
        rating = ann.feature_vals()[-1]
        #rating = (rating - 3.0)/2
    elif method is 'mean':
        all_anns = getAnnotation(meta_data, return_all=True)
        rating = np.mean([a.feature_vals() for a in all_anns], axis=(0))
    elif method is 'raw':
        all_anns = getAnnotation(meta_data, return_all=True)
        rating = [a.feature_vals() for a in all_anns]
    elif method is 'confidence':
        all_anns = getAnnotation(meta_data, return_all=True)
        rating = np.std([a.feature_vals() for a in all_anns], axis=(0))

    return np.array(rating)