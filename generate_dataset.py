import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import LIDC
from Network import data
random.seed(1337)   # for reproducibility
np.random.seed(1337)

#   Setup
# ==========

size_list = [144]
res_list  = [0.5]
norm_list = ['Normal']
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

    #LIDC.extract_from_cluster_map(cluster_map, patch_size=size, res=res, dump=do_dump)
    #LIDC.extract(patch_size=144, res="0.5I", dump=do_dump)


    filename = 'NodulePatchesNew{}-{}.p'.format(size, res)
    dataset = pickle.load(open(filename, 'br'))
    print("Loaded {} entries".format(len(dataset)))

    # post-process
    min_size = 3.0
    min_weight = 0.5
    dataset = LIDC.filter_entries(dataset, min_size=min_size, min_weight=min_weight)
    print("Filtered to {} entries, using min size = {}, and min weight = {}".format(len(dataset), min_size, min_weight))

    dataset = LIDC.append_malignancy_class(dataset)

    dataset = data.scale_image_values(dataset, window=(-1000, 400), normalize=norm)

    dataset = data.split_to_crossvalidation_groups(dataset, n_groups=5)

    out_filename = 'DatasetCV{}-{}-{}.p'.format(size, res, norm)
    pickle.dump(dataset, open('Dataset/' + out_filename, 'bw'))

    '''
    data.generate_nodule_dataset(filename='LIDC/{}ByMalignancy.p'.format(filename),
                            output_filename=out_filename,
                            test_ratio=0.2,
                            validation_ratio=0.25,
                            window=(-1000, 400),
                            normalize=norm,
                            dump=do_dump)
    '''
plt.show()
