import pickle
import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
np.random.seed(1337) # for reproducibility
from PIL import Image
#import random
#random.seed(1337)
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from Network.data import load_nodule_dataset, load_nodule_raw_dataset, normalize
from Network.dataUtils import crop_center, rating_normalize


mean_ = -517.29
std_  = 433.80


def split_by_size(size_arr, n_groups=5):
    pivots = np.arange(n_groups)[1:]*size_arr.shape[0]//n_groups
    steps = np.sort(size_arr)[np.array(pivots)]
    mapping = np.zeros(len(size_arr))
    print(steps)
    for step in steps:
        mapping = mapping + (np.array(size_arr) > step).astype('uint')
    return mapping


def scale(im):
    return (255*(im*std_ + mean_ + 1000.0)/1400.0).astype('int')

filename = 'LIDC/NodulePatches144-0.5-IByMalignancy.p'
M, B, U = pickle.load(open(filename, 'br'))

raw_dataset = load_nodule_raw_dataset(size=144, res=0.5, sample='Normal')[0]
dataset = [(crop_center(entry['patch']*(1.0+0.0*entry['mask']), entry['mask'], 128)[0], np.mean(entry['rating'], axis=0), entry['label'], entry['info'], entry['size']) for entry in raw_dataset]
#dataset += [(normalize(entry['patch'], mean_, std_, [-1000, 400])*(1.0+0.0*entry['mask']), np.mean(entry['rating'], axis=0), -1, entry['info'], entry['size']) for entry in U[:len(U)//5]]

images    = [scale(entry[0]) for entry in dataset]
rating    = [entry[1] for entry in dataset]
n_rating    = [rating_normalize(entry[1], 'Norm') for entry in dataset]
malig_map = [entry[2] for entry in dataset]
meta_data = [entry[3] for entry in dataset]
size_arr  = np.array([entry[-1] for entry in dataset])

print(np.min(n_rating, axis=0))
print(np.max(n_rating, axis=0))

# 1) Select References
# ======================

size_map = {}
for mal in np.unique(malig_map):
    size_map[mal] = split_by_size(size_arr[np.array(malig_map)==mal])
    for s in np.unique(size_map[mal]):
        cnt = np.count_nonzero((size_map[mal] == s).astype('uint'))

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

references = []
for mal in np.unique(malig_map):
    print("Maligancy: {}".format(mal))
    for s in np.unique(size_remap):
        mask = (size_remap == s).astype('uint')*(np.array(malig_map) == mal).astype('uint')
        r_c = np.random.permutation(np.where(mask)[0])[:4 if mal > -1 else 2]
        references.append(r_c)
        print(r_c)

references = np.concatenate(references)
print("selected {} references".format(references.shape[0]))
print([np.round(size_arr[r]).astype('int') for r in references])

# 2) Select Group-A
# ======================

# Find matches
nbrs = NearestNeighbors(n_neighbors=len(dataset), algorithm='auto', metric='l2').fit(n_rating)
distances, indices = nbrs.kneighbors(n_rating)

group_a = []
for ref in references:
    cand = indices[ref][1:]
    delta_size = np.abs(size_arr[ref] - size_arr[cand])
    thresh = np.maximum(np.sort(delta_size)[100], 1.5)
    select = cand[delta_size <= thresh][:10]
    group_a.append(select)

# 3) Select Group-B
# ======================
group_b = []
for i, ref in enumerate(references):
    cand = np.setdiff1d(indices[ref][1:], group_a[i])
    delta_size = np.abs(size_arr[ref] - size_arr[cand])
    #min_thresh = np.sort(delta_size)[150]
    thresh = np.maximum(np.sort(delta_size)[200], 2.5)
    select = cand[delta_size <= thresh][20:100]
    select = np.random.permutation(select)[:10]
    group_b.append(select)

# 4) Plot
# ======================
np.set_printoptions(precision=1)
ref_ids = np.array([3, 7, 12, 16, 19, 24, 29, 33, 36])
for id, ref in zip(ref_ids, references[ref_ids]):
    plt.figure()
    plt.subplot(331)
    plt.imshow(images[ref], cmap='gray')
    plt.title("R: {}".format(rating[ref]))
    plt.ylabel("S: {}".format(size_arr[ref].astype('int')))
    #print(filename + ": {}".format((np.min(images[ref]), np.max(images[ref]))))
    for i, imd_idx in enumerate(group_a[id][:4]):
        plt.subplot(3,3,i+2)
        plt.imshow(images[imd_idx], cmap='gray')
        plt.title("R: {}".format(rating[imd_idx]))
        plt.ylabel("S: {}, D: {:.2f}".format(size_arr[imd_idx].astype('int'), np.asscalar(distances[ref,indices[ref, :]==imd_idx])))
    for i, imd_idx in enumerate(group_b[id][:4]):
        plt.subplot(3,3,i+6)
        plt.imshow(images[imd_idx], cmap='gray')
        plt.title("R: {}".format(rating[imd_idx]))
        plt.ylabel("S: {}, D: {:.2f}".format(size_arr[imd_idx].astype('int'), np.asscalar(distances[ref,indices[ref, :]==imd_idx])))

# 5) Dump Images
# ======================
MAP = {} # maps filenames to meta-data
for ref_num, ref in enumerate(references):
    filename = "ref_{}.png".format(ref_num)
    Image.fromarray(images[ref]).convert('RGB').save('.\\Triplets\\' + filename)
    MAP[filename] = meta_data[ref]
    for cand_num, cand in enumerate(group_a[ref_num]):
        filename = "cand_{}_{}.png".format(ref_num, cand_num)
        Image.fromarray(images[cand]).convert('RGB').save('.\\Triplets\\' + filename)
        MAP[filename] = meta_data[cand]
    for cand_num, cand in enumerate(group_b[ref_num]):
        filename = "cand_{}_{}.png".format(ref_num, cand_num+10)
        Image.fromarray(images[cand]).convert('RGB').save('.\\Triplets\\' + filename)
        MAP[filename] = meta_data[cand]

pickle.dump(MAP, open(".\\Triplets\\triplet_map.p", "bw"))

# 6) Dump Triplets
# ======================


def ref_filename(ref_id):
    return "ref_{}.png".format(ref_id)


def cand_filename(ref_id, cand_id):
    return "cand_{}_{}.png".format(ref_id, cand_id)

configs = ["a", "b"]
for ref_num in range(len(references)):
    for c in configs:
        with open(".\\Triplets\\{}{}.txt".format(ref_num,c), "w") as text_file:
            # group-A
            text_file.write("{}, {}, {}, 0\n".format(
                        ref_filename(ref_num),
                        cand_filename(ref_num, 0),
                        cand_filename(ref_num, 1)))
            range_ = range(2, 11) if c=="a" else range(10, 1, -1)
            for i in range_:
                text_file.write("{}, {}, {}, 0\n".format(
                                    ref_filename(ref_num),
                                    cand_filename(ref_num, i),
                                    "none"))
            # group-B
            for i in range(9):
                text_file.write("{}, {}, {}, 0\n".format(
                                    ref_filename(ref_num),
                                    cand_filename(ref_num, i+10),
                                    cand_filename(ref_num, i+11)))
            text_file.write("{}, {}, {}, 0\n".format(
                ref_filename(ref_num),
                cand_filename(ref_num, 10),
                cand_filename(ref_num, 3)))
            text_file.write("{}, {}, {}, 0\n".format(
                ref_filename(ref_num),
                cand_filename(ref_num, 19),
                cand_filename(ref_num, 7)))
plt.show()