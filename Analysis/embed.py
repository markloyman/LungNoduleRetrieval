import pickle
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'E:\LungNoduleRetrieval')
from Network.dataUtils import crop_center
from Analysis.analysis import calc_embedding_statistics
import FileManager


## ===================== ##
## ======= Setup ======= ##
## ===================== ##

size        = 144
res = 'Legacy'
sample = 'Normal'

normalize = True

# 0     Test
# 1     Validation
# 2     Training
DataSubSet = 1

Weights = FileManager.Weights('siam')

wRuns = ['050', '051']
outputs = [128, 128]
in_size = 128
input_shape = (in_size, in_size, 1)
wEpchs= [22] #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]

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


## ========================= ##
## ======= Evaluate  ======= ##
## ========================= ##

start = timer()
try:

    for run, out_size in zip(wRuns, outputs):
        for epoch in wEpchs:
            filename = './embed/embed_siam{}-{}_{}.p'.format(run, epoch, post)
            try:
                images, pred, meta, labels, masks = pickle.load(open(filename, 'br'))
            except:
                from Network.data import load_nodule_dataset, load_nodule_raw_dataset, prepare_data
                from Network.model import miniXception_loader
                from Network.siameseArch import siamArch

                # prepare test data
                images, labels, masks, meta = \
                    prepare_data(load_nodule_dataset(size=size, res=res, sample=sample)[DataSubSet],
                                 categorize=False,
                                 reshuffle=False,
                                 return_meta=True,
                                 verbose=1)
                images = np.array([crop_center(im, msk, size=in_size)[0]
                                         for im, msk in zip(images, masks)])
                print("Image size changed to {}".format(images.shape))
                print('Mask not updated')

                model = siamArch(miniXception_loader, input_shape, 2, distance='l2', output_size=out_size, normalize=normalize)
                embed_model = model.extract_core(weights=Weights(run=run, epoch=epoch))
                pred = embed_model.predict(images)
                pickle.dump((images, pred, meta, labels, masks),
                            open(filename, 'bw'))
                print("Saved: {}".format(filename))

            print("Data: images({}), labels({})".format(images[0].shape, labels.shape))
            print("Range = [{:.2f},{:.2f}]".format(np.min(images[0]), np.max(images[0])))

            print(pred.shape)
            calc_embedding_statistics(pred, title=filename)

            plt.figure()
            #plt.subplot(211)
            plt.plot(np.transpose( np.squeeze(pred[np.argwhere(np.squeeze(labels==0)),
                                   np.squeeze(np.argwhere(np.std(pred, axis=0) > 0.01))])), 'blue', alpha=0.3)
            #plt.subplot(212)
            plt.plot(np.transpose( np.squeeze(pred[np.argwhere(np.squeeze(labels == 1)),
                                   np.squeeze(np.argwhere(np.std(pred, axis=0) > 0.01))])), 'red', alpha=0.2)

finally:
    plt.show()

    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))