import pickle
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, 'E:\LungNoduleRetrieval')
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

run = '029'
epoch = 29
WeightsFile =  FileManager.Weights('siam').name(run, epoch=epoch)

pred_file_format = 'embed\pred_siam{}_E{}_{}.p'
def pred_filename(run, epoch, post):
    return pred_file_format.format(run, epoch, post)

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

start = timer()
try:

    try:
        pred, labels_test, meta = pickle.load(open(pred_filename(run, epoch=epoch, post=post), 'br'))
        print("loaded saved dump of predications")
    except:
        from Network.data import load_nodule_dataset, load_nodule_raw_dataset, prepare_data_siamese
        from Network.model import miniXception_loader
        from Network.modelUtils import siamese_margin
        from Network.siameseArch import siamArch

        # prepare test data
        images_test, labels_test, masks_test, confidence, meta = \
            prepare_data_siamese(load_nodule_dataset(size=size, res=res, sample=sample)[DataSubSet], size=size,
                                 return_meta=True, verbose=1)
        print("Data ready: images({}), labels({})".format(images_test[0].shape, labels_test.shape))
        print("Range = [{:.2f},{:.2f}]".format(np.min(images_test[0]), np.max(images_test[0])))

        raw_dataset = load_nodule_raw_dataset(size=size, res=res, sample=sample)[DataSubSet]

        ## ========================= ##
        ## ======= Evaluate  ======= ##
        ## ========================= ##

        model = siamArch(miniXception_loader, input_shape, 2, distance='l2', output_size=128, normalize=True)
        if WeightsFile is not None:
            model.load_weights(WeightsFile)
            print('Load from: {}'.format(WeightsFile))
        else:
            print('Model without weights')

        print("Begin Predicting...")
        pred = model.predict(images_test, round=False)
        print("Predication Ready")

        pickle.dump((pred, labels_test, meta), open(pred_filename(run, epoch=epoch, post=post), 'bw'))

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