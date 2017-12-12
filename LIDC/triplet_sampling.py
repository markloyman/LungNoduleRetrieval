import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from Network.data import load_nodule_dataset, load_nodule_raw_dataset
from LIDC.lidcUtils import calc_rating


def split_by_size(size_arr, n_groups=5):
    pivots = np.arange(n_groups)[1:]*size_arr.shape[0]//n_groups
    steps = np.sort(size_arr)[np.array(pivots)]
    mapping = np.zeros(len(size_arr))
    print(steps)
    for step in steps:
        mapping = mapping + (np.array(size_arr) > step).astype('uint')
    return mapping


filename = 'LIDC/NodulePatches144-LegacyByMalignancy.p'
M, B, U = pickle.load(open(filename, 'br'))

raw_dataset = load_nodule_raw_dataset(size=144, res='Legacy', sample='Normal')[0]
dataset = [(entry['patch']*(0.3+0.7*entry['mask']),  np.mean(entry['rating'], axis=0), entry['label'], entry['info'], entry['size']) for entry in raw_dataset]
dataset += [(entry['patch']*(0.3+0.7*entry['mask']), np.mean(entry['rating'], axis=0), -1, entry['info'], entry['size']) for entry in U[:len(U)//5]]

images    = [entry[0] for entry in dataset]
rating    = [entry[1] for entry in dataset]
malig_map = [entry[2] for entry in dataset]
size_arr  = np.array([entry[-1] for entry in dataset])

#size_map  = split_by_size(size_arr)
#for mal in np.unique(malig_map):
#    print("Maligancy: {}".format(mal))
#    for s in np.unique(size_map):
#        cnt = np.count_nonzero((malig_map == mal).astype('uint')*(size_map == s).astype('uint'))
#        print(cnt)

# Select References
size_map = {}
for mal in np.unique(malig_map):
    #print("Maligancy: {}".format(mal))
    size_map[mal] = split_by_size(size_arr[np.array(malig_map)==mal])
    for s in np.unique(size_map[mal]):
        cnt = np.count_nonzero((size_map[mal] == s).astype('uint'))
        #print(cnt)

size_remap = np.zeros(size_arr.shape)
for mal in np.unique(malig_map):
    idx = np.array(malig_map)==mal
    size_remap[idx] = size_map[mal]

for mal in np.unique(malig_map):
    print("Maligancy: {}".format(mal))
    for s in np.unique(size_remap):
        mask = (size_remap == s).astype('uint')*(np.array(malig_map) == mal).astype('uint')
        cnt = np.count_nonzero(mask)
        print(cnt)
ref_candidates = []
for mal in np.unique(malig_map):
    print("Maligancy: {}".format(mal))
    for s in np.unique(size_remap):
        mask = (size_remap == s).astype('uint')*(np.array(malig_map) == mal).astype('uint')
        r_c = np.random.permutation(np.where(mask)[0])[:4 if mal > -1 else 2]
        ref_candidates.append(r_c)
        print(r_c)

# Find matches
nbrs = NearestNeighbors(n_neighbors=len(dataset), algorithm='auto', metric='cosine').fit(rating)
distances, indices = nbrs.kneighbors(rating)

for r_c in ref_candidates:
    for ref in r_c[:2]:
        cand = indices[ref][1:]
        delta_size = np.abs(size_arr[ref] - size_arr[cand])
        thresh = np.sort(delta_size)[100]
        select = cand[delta_size <= thresh][:8]
        plt.figure()
        plt.subplot(331)
        plt.imshow(images[ref], cmap='gray')
        plt.title("S: {}, R: {}".format(size_arr[ref].astype('int'), rating[ref].astype('int')))
        for i, imd_idx in enumerate(select):
            plt.subplot(3,3,i+2)
            plt.imshow(images[imd_idx], cmap='gray')
            plt.title("S: {}, R: {}".format(size_arr[imd_idx].astype('int'), rating[imd_idx].astype('int')))

