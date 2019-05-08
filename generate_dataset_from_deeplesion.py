import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import LIDC
import DeepLesion

from Network import dataset as data
# for reproducibility
random.seed(1337)
np.random.seed(1337)

#   Setup
# ==========

data_path = 'DeepLesion\data'
meta_path = 'DeepLesion'
res  = 0.5
size = 160
norm = 'Normal'  # UniformNC Uniform Normal
n_groups = 5
do_dump = False

# ===================================
#   Extract ("Raw") Data from DeepLesion
# ===================================

dataset = DeepLesion.extract(data_path=data_path, meta_path=meta_path, patch_size=size, res=res)
#pickle.dump(dataset, open(filename, 'wb'))


# ===================================
#   Post-Process dataset
# ===================================

#TODO: use LIDC statistics

window = (-1000, 400)
dataset = data.scale_image_values(dataset, window=window, normalize=norm) # statistics=(-500, 420)
print("Rescaled images with window={} and {} normalization".format(window, norm))

dataset = data.split_to_crossvalidation_groups(dataset, n_groups=n_groups)
print("Entries split to {} groups".format(len(dataset)))

for i, group in enumerate(dataset):
    out_filename = 'DatasetFullCV{}_{}-{}-{}.p'.format(i, size, res, norm)
    pickle.dump(group, open('Dataset/' + out_filename, 'bw'))
    print("Dumped to {}".format(out_filename))

plt.show()
