import numpy as np
import matplotlib.pyplot as plt
import pickle
from timeit import default_timer as timer

from LIDC.lidcUtils import getAnnotation
from analysis import MalignancyConfusionMatrix, MalignancyBySize
from data import load_nodule_dataset, load_nodule_raw_dataset, prepare_data
from model import miniXception_loader
from directArch import directArch
from siameseArch import siamArch

import FileManager

## ===================== ##
## ======= Setup ======= ##
## ===================== ##

size        = 128
input_shape = (size,size,1)

# 0     Test
# 1     Validation
# 2     Training
DataSubSet = 1

Weights = FileManager.Weights('siam')

wRuns = ['000'] #, '001']
wEpchs= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]

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

dataset = load_nodule_raw_dataset()[DataSubSet]
print("Raw Data Loaded: {} entries".format(len(dataset)))

# prepare test data
images_test, labels_test, meta_test = prepare_data( load_nodule_dataset()[DataSubSet],
                                                    size        = size,
                                                    categorize  = False,
                                                    reshuffle   = False,
                                                    return_meta = True,
                                                    verbose     = 1 )

print("Data ready: images({}), labels({})".format(images_test[0].shape, labels_test.shape))
print("Range = [{},{}]".format(np.min(images_test[0]), np.max(images_test[0])))

## ========================= ##
## ======= Evaluate  ======= ##
## ========================= ##

model = siamArch(miniXception_loader, input_shape, 2)

start = timer()
try:

    for run in wRuns:
        for epoch in wEpchs:

            #model.load_weights(Weights(run=run, epoch=epoch))
            embed_model = model.extract_core(weights=Weights(run=run, epoch=epoch))

            pred = embed_model.predict(images_test)
            #print(pred.shape)

            filename = './embed/embed_siam{}-{}_{}.p'.format(run, epoch, post)
            pickle.dump(    (images_test, pred, meta_test, labels_test),
                            open(filename,'bw') )
            print("Saved: {}".format(filename))

finally:

    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))