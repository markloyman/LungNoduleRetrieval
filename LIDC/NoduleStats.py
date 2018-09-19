import pylidc as pl
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats
from LIDC.lidcUtils import getNoduleSize

plt.interactive(False)

'''
Feature              Meaning                    # 
-                    -                          - 
Subtlety           | Fairly Subtle            | 3   5
Internalstructure  | Soft Tissue              | 1   4
Calcification      | Absent                   | 6   6
Sphericity         | Ovoid/Round              | 4   5
Margin             | Medium Margin            | 3   5
Lobulation         | No Lobulation            | 1   5
Spiculation        | No Spiculation           | 1   5
Texture            | Solid                    | 5   5
Malignancy         | Moderately Unlikely      | 2   5
'''




# ----- Main -----

def show_nodule_size_and_pixel_size():
    nodSize  = []
    pixSpace = []
    for scan in pl.query(pl.Scan).all():
    # cycle 1018 scans
        nods = scan.cluster_annotations(metric='jaccard', tol=0.95, tol_limit=0.7)
        if len(nods) > 0:
            print("Scan of patient {}: {} nodules.".format(scan.patient_id,len(nods)))
            nodSize  = nodSize  + [getNoduleSize(nod) for nod in nods]
            pixSpace = pixSpace + [scan.pixel_spacing]

    size = np.array(nodSize)
    np.save('NoduleSize.npy',size)

    pixSpace = np.array(pixSpace)
    np.save('PixSpace.npy',pixSpace)

    plt.figure(1)
    plt.hist(size)
    plt.title("Nodule Size. Total of {} nodules found.".format(len(size)))

    plt.figure(2)
    plt.hist(pixSpace)
    plt.title("Pixel Spacing. Total of {} scans.".format(len(pixSpace)))

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

    distribution = np.bincount(r[:,2])[1:] + 1e-9
    distribution /= np.sum(distribution)
    entropy = -distribution.dot(np.log2(distribution) / np.log2(max_range))
    return entropy

def show_all_stats(filename):
    dataset = pickle.load(open(filename, 'br'))
    elements = ['Subtlety', 'Internalstructure', 'Calcification', 'Sphericity', 'Margin', 'Lobulation', 'Spiculation', 'Texture', 'Malignancy']

    for e,i in zip(elements, range(len(elements))):
        entropy = stat_analyze(dataset, elementID=i, title=e)
        print('{}: {:.2f}'.format(e, entropy))

if __name__ == "__main__":

    filename  = 'NodulePatchesNew160-0.5.p' # Dataset/DatasetPrimaryCV0_160-0.5-Normal.p

    show_all_stats(filename)
    #show_nodule_size_and_pixel_size()

    plt.show()
