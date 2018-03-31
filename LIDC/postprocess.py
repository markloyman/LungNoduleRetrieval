import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from LIDC.NoduleStats import stat_analyze

plt.interactive(False)


def vote(rating):
#   0:  benign
#   1:  malignant
#   2: unknown
    b = np.count_nonzero(rating < 3,  axis=0)
    m = np.count_nonzero(rating > 3,  axis=0)
    u = np.count_nonzero(rating == 3, axis=0)
    v = np.argmax(np.vstack([b,m,u]), axis=0)
    v = np.where(m==b,2, v)
    return v


def mode_round(rating):
    rating = np.where(rating == 4, 5, rating)
    rating = np.where(rating == 2, 1, rating)
    return stats.mode(rating, axis=0)[0]


def weighted_sum(rating):
    rating = rating - 3
    rating = np.sum(rating, axis=0)
    rat = np.where(rating>0, 5, 0)
    rat = np.where(rating<0, 1, rat)
    return rat


def append_label(data, label, confidence_level = None):
    new_data = []
    for entry in data:
        entry['label'] = label
        if confidence_level is not None:
            entry['confidence'] = confidence_level
        new_data.append(entry)

    return new_data


def append_and_group_malignancy_class_to_nodule_db(filename, save_dump = False):

    dataset = pickle.load(open(filename,'br'))

    Malig = [ vote(np.array([r[8] for r in entry['rating']])) for entry in dataset]

    lenM = np.count_nonzero(np.array(Malig)==1)
    lenB = np.count_nonzero(np.array(Malig)==0)
    lenU = np.count_nonzero(np.array(Malig)==2)

    print("{} malignant, {} benign and {} unknown".format(lenM, lenB, lenU))

    M = np.extract(np.squeeze(np.array(Malig) == 1), dataset)
    B = np.extract(np.squeeze(np.array(Malig) == 0), dataset)
    U = np.extract(np.squeeze(np.array(Malig) == 2), dataset)

    append_label(M, 1)
    append_label(B, 0)
    append_label(U, -1)

    stat_analyze(M, 8, 'Maligancy Statistics')
    stat_analyze(B, 8, 'Benign Statistics')
    stat_analyze(U, 8, 'Undetermied Statistics')


    if save_dump:
        pickle.dump((M,B,U), open(filename[:-2] + 'ByMalignancy.p','bw'))

    outliers_M = np.where(np.array([np.min(m['rating'], axis=(0))[-1] for m in M]) < 2)[0]
    outliers_B = np.where(np.array([np.max(b['rating'], axis=(0))[-1] for b in B]) > 4)[0]
    print('Outliers M: {}'.format(outliers_M))
    print('Outliers B: {}'.format(outliers_B))


def append_malignancy_class(dataset):
    for entry in dataset:
        entry['label'] = vote(np.array([r[8] for r in entry['rating']]))

    lenM = np.count_nonzero([entry['label']==1 for entry in dataset])
    lenB = np.count_nonzero([entry['label']==0 for entry in dataset])
    lenU = np.count_nonzero([entry['label']==2 for entry in dataset])
    print("{} malignant, {} benign and {} unknown".format(lenM, lenB, lenU))

    return dataset


def entry_is_valid(entry, min_size, min_weight):
    size_condition = np.max(entry['ann_size']) > min_size
    weight_condition = np.max(entry['weights']) > min_weight
    return  size_condition and weight_condition


def filter_entries(dataset, min_size, min_weight):
    return list(filter(lambda x: entry_is_valid(x, min_size, min_weight), dataset))


if __name__ == "__main__":


    #append_malignancy_class_to_nodule_db('NodulePatches128-Legacy.p', save_dump=True)         DONE
    #append_malignancy_class_to_nodule_db('NodulePatches144-0.5-I.p', save_dump=True)
    #append_malignancy_class_to_nodule_db('NodulePatches128-0.7.p', save_dump=True)
    #append_malignancy_class_to_nodule_db('NodulePatches144-0.5.p', save_dump=True)         YET

    plt.show()
