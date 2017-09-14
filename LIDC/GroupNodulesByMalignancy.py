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


def append_malignancy_class_to_nodule_db(filename, save_dump = False):

    dataset = pickle.load(open(filename,'br'))

    Malig = [ vote(np.array([r[8] for r in entry['rating']])) for entry in dataset]

    lenM = np.count_nonzero(np.array(Malig)==1)
    lenB = np.count_nonzero(np.array(Malig)==0)
    lenU = np.count_nonzero(np.array(Malig)==2)

    print("{} malignant, {} benign and {} unknown".format(lenM, lenB, lenU))

    M = np.extract(np.reshape(np.array([np.array(Malig)==1])[:], len(dataset)), dataset)
    B = np.extract(np.reshape(np.array([np.array(Malig)==0])[:], len(dataset)), dataset)
    U = np.extract(np.reshape(np.array([np.array(Malig)==2])[:], len(dataset)), dataset)

    append_label(M, 1)
    append_label(B, 0)

    stat_analyze(M, 8, 'Maligancy Statistics')
    stat_analyze(B, 8, 'Benign Statistics')
    stat_analyze(U, 8, 'Undetermied Statistics')


    if save_dump:
        pickle.dump((M,B,U), open(filename[:-2] + 'ByMalignacy.p','bw'))

    outliers_M = np.where(np.array([np.min(m['rating'], axis=(0))[-1] for m in M]) < 2)[0]
    outliers_B = np.where(np.array([np.min(b['rating'], axis=(0))[-1] for b in B]) > 4)[0]
    print('Outliers M: {}'.format(outliers_M))
    print('Outliers B: {}'.format(outliers_B))


if __name__ == "__main__":

    append_malignancy_class_to_nodule_db('NodulePatchesClique.p', save_dump=False)

    plt.show()
