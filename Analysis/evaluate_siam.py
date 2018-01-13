import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
np.random.seed(1337)  # for reproducibility
import random
random.seed(1337)
import pickle
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'E:\LungNoduleRetrieval')
import FileManager
from Network.dataUtils import crop_center

eps = 0.0000000001


def find_full_entry(query, raw):
    for entry in raw:
        entry_meta = entry['info']
        if  query[0] == entry_meta[0] and \
                query[1] == entry_meta[1] and \
                query[2] == entry_meta[2] and \
                query[3] == entry_meta[3]:
            return entry
    return None


def precision(true, pred, thresh):
    true = true.astype('int')
    pred = (pred > thresh).astype('int')
    TP = np.count_nonzero( (1-true) * (1-pred) )
    FP = np.count_nonzero( true * (1-pred) )
    return (TP+eps) / (TP+FP+eps)


def recall(true, pred, thresh):
    true = true.astype('int')
    pred = (pred > thresh).astype('int')
    TP = np.count_nonzero((1 - true) * (1 - pred))
    FN = np.count_nonzero( (1-true) * pred)
    return (TP+eps) / (TP+FN+eps)

## ===================== ##
## ======= Setup ======= ##
## ===================== ##

size        = 144
input_shape = (128,128,1)
sample = 'Normal'
res = '0.5I'

in_size = 128
out_size = 128
normalize = True

load     = False
evaluate = False
force = False

# 0     Test
# 1     Validation
# 2     Training
DataSubSet = 2

run = '000'
epoch = 5
WeightsFile =  FileManager.Weights('siamR').name(run, epoch=epoch)

pred_file_format = '.\output\embed\pred_siam{}_E{}_{}.p'
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
        if force is True:
            open('junk', 'br')
        pred, labels_test, meta = pickle.load(open(pred_filename(run, epoch=epoch, post=post), 'br'))
        print("loaded saved dump of predications")
        from Network.data import load_nodule_raw_dataset
        from Network.metrics import siamese_margin
    except:
        from Network.data import load_nodule_dataset, load_nodule_raw_dataset, prepare_data_siamese, prepare_data_siamese_simple
        from Network.model import miniXception_loader
        from Network.metrics import siamese_margin
        from Network.siameseArch import siamArch

        # prepare model
        #model = siamArch(miniXception_loader, input_shape, 2, distance='l2', output_size=out_size, normalize=normalize)
        model = siamArch(miniXception_loader, input_shape, objective="rating", distance='l2', pooling="rmac", output_size=out_size, normalize=normalize)
        if WeightsFile is not None:
            model.load_weights(WeightsFile)
            print('Load from: {}'.format(WeightsFile))
        else:
            print('Model without weights')

        pred_all = []
        labels_test_all = []
        meta_all_0 = []
        meta_all_1 = []
        for i in range(1):
            # prepare test data
            images_test, labels_test,  masks_test, confidence, meta = \
                prepare_data_siamese_simple(load_nodule_dataset(size=size, res=res, sample=sample)[DataSubSet], size=size,
                                     return_meta=True, objective="rating", verbose=1, balanced=True)
            print("Data ready: images({}), labels({})".format(images_test[0].shape, labels_test.shape))
            print("Range = [{:.2f},{:.2f}]".format(np.min(images_test[0]), np.max(images_test[0])))

            images_test = (np.array([crop_center(im, msk, size=in_size)[0]
                                for im, msk in zip(images_test[0], masks_test[0])]),
                      np.array([crop_center(im, msk, size=in_size)[0]
                                for im, msk in zip(images_test[1], masks_test[1])]))
            print("Image size changed to {}".format(images_test[0].shape))
            print('Mask not updated')

            # eval
            print("Begin Predicting...")
            pred = model.predict(images_test, round=False)
            print("Predication Ready")

            pred_all.append(pred)
            labels_test_all.append(labels_test)
            meta_all_0 += meta[0]
            meta_all_1 += meta[1]
        pred = np.concatenate(pred_all)
        labels_test = np.concatenate(labels_test_all)
        meta = (meta_all_0, meta_all_1)
        pickle.dump((pred, labels_test, meta), open(pred_filename(run, epoch=epoch, post=post), 'bw'))


    ## ========================= ##
    ## ======= Evaluate  ======= ##
    ## ========================= ##

    if evaluate:

        raw_dataset = load_nodule_raw_dataset(size=size, res=res, sample=sample)[DataSubSet]

        failed_same, failed_diff = [], []
        K = 12

        p = np.zeros(K)
        r = np.zeros(K)
        for t in range(K):
            p[t] = precision(labels_test, np.squeeze(pred), 0.1*t)
            r[t] = recall(labels_test, np.squeeze(pred), 0.1*t)
        plt.figure()
        plt.title('PR Curve')
        plt.plot(r, p)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.axis([0, 1, 0, 1])

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
                    failed_diff.append((m1_entry, m2_entry, p))

            else:
                # same
                s.append(p)
                if p > 0.5*siamese_margin:
                    m1_entry = find_full_entry(m1, raw_dataset)
                    m2_entry = find_full_entry(m2, raw_dataset)
                    failed_same.append((m1_entry, m2_entry, p))

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

        pickle.dump((failed_same, failed_diff), open('eval_dump_{}e{}.p'.format(run, epoch), 'bw'))

        print('Plots Ready...')
        plt.show()

finally:
    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))