import matplotlib.pyplot as plt
import numpy as np
from Network.dataUtils import augment, crop_center
import sys
sys.path.insert(0, 'E:\LungNoduleRetrieval')

from Network.data import load_nodule_dataset, load_nodule_raw_dataset

def find_full_entry(query, raw):
    id=0
    for entry in raw:
        entry_meta = entry[3]
        if  query[0] == entry_meta[0] and \
                query[1] == entry_meta[1] and \
                query[2] == entry_meta[2] and \
                query[3] == entry_meta[3]:
            print(id)
            return entry
        id = id+1
    return None


data128 = load_nodule_dataset(128, res='Legacy', sample='Normal', apply_mask_to_patch=True)
data128 = data128[0] + data128[1] + data128[2]

data144 = load_nodule_dataset(144, res=0.7, sample='Normal', apply_mask_to_patch=True)
data144 = data144[0] + data144[1] + data144[2]

im_id = 1105
match = find_full_entry(data128[im_id][3], data144)

plt.figure()
plt.subplot(131)
plt.imshow(data128[im_id][0])
plt.subplot(132)
#plt.imshow(crop_center(match[0], match[1], size=128)[0])
new_im = augment(match[0], match[1], size=128, crop_stdev=0.0)[0]
plt.imshow(new_im)
plt.subplot(133)
diffff = new_im-data128[im_id][0]
plt.imshow(diffff)
plt.title(np.sum(np.square(diffff)))
