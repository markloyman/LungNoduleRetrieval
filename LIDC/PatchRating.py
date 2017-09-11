import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

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


def stat_analyze(dataset, elementID, title):
    #labels = ['max', 'min', 'median', 'mode', 'sum']
    #assert elementID==8 # otherwise range needs to be corrected
    labels = ['max', 'min', 'median', 'mode']
    Rating = [(np.max(entry['rating'], axis=0),
               np.min(entry['rating'], axis=0),
               np.median(entry['rating'], axis=0).astype('uint'),
               stats.mode(entry['rating'], axis=0)[0]
               #weighted_sum(entry['rating'])
               )
              for entry in dataset]
    r   = np.array([ np.array([  r[0][elementID],       # max
                                 r[1][elementID],       # min
                                 r[2][elementID],       # median
                                 r[3][0][elementID]     # mode
                                 #r[4][elementID]
                                ]) for r in Rating])
    max_range = np.max(r)
    plt.figure(title)
    for i in range(r.shape[1]):
        plt.subplot(r.shape[1],1,int(i+1))
        arr = plt.hist(r[:,i], bins=max_range, range=(0.5, 0.5+max_range))
        for k in range(max_range):
            plt.text(arr[1][k]+0.5, arr[0][k], str(arr[0][k]))
        plt.xlim(0.5, 0.5+max_range)
        plt.title(labels[i])

def append_label(data, label, confidence_level = None):
    new_data = []
    for entry in data:
        entry['label'] = label
        if confidence_level is not None:
            entry['confidence'] = confidence_level
        new_data.append(entry)

    return new_data


def append_maligancy_class_to_nodule_db(filename, save_dump = False):

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

    stat_analyze(M,8, 'Maligancy Statistics')
    stat_analyze(B,8, 'Benign Statistics')
    stat_analyze(U,8, 'Undetermied Statistics')


    if save_dump:
        pickle.dump((M,B,U), open(filename[:-2] + 'ByMalignacy.p','bw'))


def show_all_stats(filename):
    dataset = pickle.load(open(filename, 'br'))

    elements = ['Subtlety', 'Internalstructure', 'Calcification', 'Sphericity', 'Margin', 'Lobulation', 'Spiculation', 'Texture', 'Malignancy']

    for e,i in zip(elements, range(len(elements))):
        stat_analyze(dataset, elementID=i, title=e)

if __name__ == "__main__":
    filename  = 'NodulePatches.p'
    #append_maligancy_class_to_nodule_db(filename)
    show_all_stats(filename)
    plt.show()
