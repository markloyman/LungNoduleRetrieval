import numpy as np
import matplotlib.pyplot as plt
import pickle
from timeit import default_timer as timer

from LIDC.lidcUtils import getAnnotation
from analysis import MalignancyConfusionMatrix, MalignancyBySize
from data import load_nodule_dataset, load_nodule_raw_dataset, prepare_data_siamese_chained
from model import miniXception_loader
from directArch import directArch
from siameseArch import siamArch


## ===================== ##
## ======= Setup ======= ##
## ===================== ##

size        = 128
input_shape = (128,128,1)

load     = False
evaluate = False

# 0     Test
# 1     Validation
# 2     Training
DataSubSet = 0


run = '000'
WeightsFile = 'w_siam000_15-3.55-4.77.h5'
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
images_test, labels_test = prepare_data_siamese_chained(load_nodule_dataset()[DataSubSet], size=size, return_meta=False)
print("Data ready: images({}), labels({})".format(images_test[0].shape, labels_test.shape))
print("Range = [{},{}]".format(np.min(images_test[0]), np.max(images_test[0])))

## ========================= ##
## ======= Evaluate  ======= ##
## ========================= ##

model = siamArch(miniXception_loader, input_shape, 2)
model.load_weights(WeightsFile)


start = timer()
try:

    if load:
        pred, Size = pickle.load(open('embed\pred_siam{}_{}.p'.format(run, post),'br'))
        print("loaded saved dump of predications")
    else:
        #Size = [getAnnotation(entry['info']).estimate_diameter() for entry in dataset]
        print("Begin Predicting...")
        pred = model.predict(images_test, round=False)
        print("Predication Ready")

        pickle.dump((pred, None),open('embed\pred_siam{}_{}.p'.format(run, post),'bw'))


    if evaluate:
        d,s = [], []
        for l,p in zip(labels_test, pred): # similarity labels
            if l:
                # different
                d.append(p)
            else:
                # same
                s.append(p)

        plt.subplot(121)
        plt.hist(np.array(s))
        plt.title("Distance histogram for Same")

        plt.subplot(122)
        plt.hist(np.array(d))
        plt.title("Distance histogram for Different")

        print('Plots Ready...')
        plt.show()

finally:
    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))