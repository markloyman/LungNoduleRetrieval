import pickle
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, 'E:\LungNoduleRetrieval')

from Network.data import load_nodule_dataset, load_nodule_raw_dataset, prepare_data_siamese
from Network.model import miniXception_loader
from Network.modelUtils import siamese_margin
from Network.siameseArch import siamArch
from Analysis.analysis import calc_embedding_statistics
import FileManager


def find_full_entry(query, raw):
    for entry in raw:
        entry_meta = entry['info']
        if  query[0] == entry_meta[0] and \
                query[1] == entry_meta[1] and \
                query[2] == entry_meta[2] and \
                query[3] == entry_meta[3]:
            return entry
    return None


## ===================== ##
## ======= Setup ======= ##
## ===================== ##

size        = 128
input_shape = (128,128,1)
sample = 'Normal'
res = 'Legacy'

load     = False
evaluate = True

# 0     Test
# 1     Validation
# 2     Training
DataSubSet = 1


run = '020'
#WeightsFile = 'Weights/w_siam00YY_19-5.60-12.42.h5'
#WeightsFile = 'Weights/w_siam015_40-0.08-4.56.h5'
WeightsFile = FileManager.Weights('siam').name('020', epoch=40)
#WeightsFile = 'Weights/w_siam00Z_00-52.79-0.34.h5'
#WeightsFile = 'Weights/w_siam00Z_14-0.25-0.32.h5'
#WeightsFile = 'Weights/w_siam00Y_14-0.69-0.87.h5'
#WeightsFile = 'w_siam000_25-0.46-4.77.h5'
#WeightsFile = 'w_siam001_30-1.78-5.15.h5'
#WeightsFile = 'w_siam001_40-0.49-4.73.h5'

## ========================= ##
## ======= Load Data ======= ##
## ========================= ##

if   DataSubSet == 0:
    post = "Test"
elif DataSubSet == 1:
    post = "Valid"
elif DataSubSet == 2:
    post = "Train"
else:
    assert False
print("{} Set Analysis".format(post))
print('='*15)

#dataset = load_nodule_raw_dataset()[DataSubSet]
#print("Raw Data Loaded: {} entries".format(len(dataset)))

# prepare test data
images_test, labels_test, masks_test, confidence, meta = \
    prepare_data_siamese(load_nodule_dataset(size=size, res=res, sample=sample)[DataSubSet], size=size, return_meta=True, verbose=1)
print("Data ready: images({}), labels({})".format(images_test[0].shape, labels_test.shape))
print("Range = [{:.2f},{:.2f}]".format(np.min(images_test[0]), np.max(images_test[0])))


raw_dataset = load_nodule_raw_dataset(size=size, res=res, sample=sample)[DataSubSet]

## ========================= ##
## ======= Evaluate  ======= ##
## ========================= ##

model = siamArch(miniXception_loader, input_shape, 2, distance='l2')
if WeightsFile is not None:
    model.load_weights(WeightsFile)
    print('Load from: {}'.format(WeightsFile))
else:
    print('Model without weights')

start = timer()
try:

    if load:
        pred, Size = pickle.load(open('embed\pred_siam{}_{}.p'.format(run, post), 'br'))
        print("loaded saved dump of predications")
    else:
        #Size = [getAnnotation(entry['info']).estimate_diameter() for entry in dataset]
        print("Begin Predicting...")
        pred = model.predict(images_test, round=False)
        print("Predication Ready")

        pickle.dump((pred, None), open('embed\pred_siam{}_{}.p'.format(run, post), 'bw'))

    calc_embedding_statistics(pred)

    failed_same, failed_diff = [], []
    if evaluate:
        diff_len = np.count_nonzero(labels_test)
        same_len = len(labels_test) - diff_len

        d, s = [], []
        for l, p, m1, m2 in zip(labels_test, pred, meta[0], meta[1]): # similarity labels
            if l:
                # different
                d.append(p)
                if p < 0.5*siamese_margin:
                    m1_entry = find_full_entry(m1, raw_dataset)
                    m2_entry = find_full_entry(m2, raw_dataset)
                    failed_diff.append((m1_entry, m2_entry))

            else:
                # same
                s.append(p)
                if p > 0.5*siamese_margin:
                    m1_entry = find_full_entry(m1, raw_dataset)
                    m2_entry = find_full_entry(m2, raw_dataset)
                    failed_same.append((m1_entry, m2_entry))

        plt.figure('{}-{}'.format(run, post))

        plt.subplot(121)
        plt.hist(np.array(s))
        plt.title("Same, Err: {:.0f}/{:.0f} ({:.1f})".format(len(failed_same), same_len, len(failed_same)/same_len))
        plt.xlabel('Distance')
        plt.ylabel('Histogram')

        plt.subplot(122)
        plt.hist(np.array(d))
        plt.title("Diff, Err: {:.0f}/{:.0f} ({:.1f})".format(len(failed_diff), diff_len, len(failed_diff)/diff_len))
        plt.xlabel('Distance')
        plt.ylabel('Histogram')

        pickle.dump((failed_same, failed_diff), open('eval_dump.p', 'bw'))

        print('Plots Ready...')
        plt.show()

finally:
    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))