import matplotlib.pyplot as plt
import numpy as np
import random
import LIDC
from Network.data import generate_nodule_dataset
random.seed(1337)   # for reproducibility
np.random.seed(1337)

#   Setup
# ==========

size_list = [144]
res_list  = ['0.5I']
norm_list = ['Normal']
do_dump = False

assert(len(size_list) == len(res_list))
assert(len(size_list) == len(norm_list))

#   Run
# ==========

for size, res, norm in zip(size_list, res_list, norm_list):

    LIDC.extract(patch_size=144, res="0.5I", dump=do_dump)

    filename = 'NodulePatches{}-{}.p'.format(size, res)
    LIDC.append_malignancy_class_to_nodule_db(filename, save_dump=do_dump)

    out_filename = 'Dataset{}-{}-{}.p'.format(size, res, norm)
    generate_nodule_dataset(filename='LIDC/{}ByMalignancy.p'.format(filename),
                            output_filename=out_filename,
                            test_ratio=0.2,
                            validation_ratio=0.25,
                            window=(-1000, 400),
                            normalize=norm,
                            dump=do_dump)

plt.show()