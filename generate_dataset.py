import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import LIDC
from Network import dataset as data
random.seed(1337)   # for reproducibility
np.random.seed(1337)

#   Setup
# ==========

size_list = [144]*2
res_list  = [0.5, 0.7]
norm_list = ['Normal']*2  # UniformNC Uniform Normal
n_groups = 5
do_dump = True

assert(len(size_list) == len(res_list))
assert(len(size_list) == len(norm_list))

#   Run
# ==========

clusters_filename = "LIDC/cluster_map.p"
try:
    cluster_map = pickle.load(open(clusters_filename, 'br'))
except:
    cluster_map = LIDC.cluster_all_annotations()
    pickle.dump(cluster_map, open(clusters_filename, 'bw'))

for size, res, norm in zip(size_list, res_list, norm_list):

    # ===================================
    #   Extract ("Raw") Data from LIDC
    # ===================================

    if False:
        filename = 'LIDC/NodulePatchesNew{}-{}.p'.format(size, res)
        try:
            dataset = pickle.load(open(filename, 'br'))
        except:
            dataset = LIDC.extract_from_cluster_map(cluster_map, patch_size=size, res=res)
            # LIDC.extract(patch_size=144, res="0.5I", dump=do_dump)
            pickle.dump(dataset, open(filename, 'wb'))
            print("Dumped to {}".format(filename))
        print("Loaded {} entries from {}".format(len(dataset), filename))
    else:
        dataset = [None]*5

    # ===================================
    #   Post-Process dataset
    # ===================================

    try:
        for i in range(n_groups):
            out_filename = 'DatasetFullCV{}_{}-{}-{}.p'.format(i, size, res, norm)
            dataset[i] = pickle.load(group, open('Dataset/' + out_filename, 'br'))
            print("Loaded from {}".format(out_filename))

    except:
        min_size = 3.0
        min_weight = 0.5
        dataset = LIDC.filter_entries(dataset, min_size=min_size, min_weight=min_weight)
        print("Filtered to {} entries, using min size = {}, and min weight = {}".format(len(dataset), min_size, min_weight))

        # check masks size
        masks = np.concatenate([np.expand_dims(e['mask'], axis=0) for e in dataset], axis=0)
        mask_sizes = [np.max([np.max(a) - np.min(a) for a in np.nonzero(m)]) for m in masks]
        plt.figure()
        plt.title("{} mask size".format(out_filename))
        plt.hist(mask_sizes, 20)

        dataset = LIDC.append_malignancy_class(dataset)
        print("Appended malignancy class")

        window = (-1000, 400)
        dataset = data.scale_image_values(dataset, window=window, normalize=norm) # statistics=(-500, 420)
        print("Rescaled images with window={} and {} normalization".format(window, norm))

        dataset = data.split_to_crossvalidation_groups(dataset, n_groups=n_groups)
        print("Entries split to {} groups".format(len(dataset)))

        for i, group in enumerate(dataset):
            out_filename = 'DatasetFullCV{}_{}-{}-{}.p'.format(i, size, res, norm)
            pickle.dump(group, open('Dataset/' + out_filename, 'bw'))
            print("Dumped to {}".format(out_filename))

    # ===================================
    #   Derive secondary datasets
    # ===================================
    plt.figure()
    for i, group in enumerate(dataset):
        #
        #   filter to Primary slices
        #
        print("Group #{}:".format(i))
        group = data.filter_to_primary(group)
        label_counter = np.bincount(np.concatenate([e['label'] for e in group]))
        print("\tFiltered to {} primary entries".format(len(group)))
        print("\t{} benign, {} malignant, {} unknown".format(label_counter[0], label_counter[1], label_counter[2]))

        out_filename = 'DatasetPrimaryCV{}_{}-{}-{}.p'.format(i, size, res, norm)
        pickle.dump(group, open('Dataset/' + out_filename, 'bw'))
        print("\tDumped to {}".format(out_filename))

        #
        #   crop all
        #
        new_size = 128
        group = data.crop_dataset(group, size=new_size)
        print("\tpatch size = {}".format(group[0]['patch'].shape))

        out_filename = 'DatasetPrimaryCV{}_{}-{}-{}.p'.format(i, new_size, res, norm)
        pickle.dump(group, open('Dataset/' + out_filename, 'bw'))
        print("\tDumped to {}".format(out_filename))

        # check masks
        masks = np.concatenate([np.expand_dims(e['mask'], axis=0) for e in group], axis=0)
        mask_sizes = [np.max([np.max(a) - np.min(a) for a in np.nonzero(m)]) for m in masks]
        plt.subplot(n_groups, 1, i)
        plt.title("{} mask size".format(out_filename))
        plt.hist(mask_sizes, 20)

plt.show()
