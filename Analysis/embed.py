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

network = 'siamR'

size        = 144
res = '0.5I' #'Legacy'
sample = 'Normal'

normalize = True

# 0     Test
# 1     Validation
# 2     Training
DataSubSet = 2

Weights = FileManager.Weights(network)
Embed   = FileManager.Embed(network)

wRuns = ['006XX'] #['064X', '071' (is actually 071X), '078X', '081', '082']

outputs = [128]*len(wRuns)
in_size = 128
input_shape = (in_size, in_size, 1)
wEpchs= [5, 10, 15, 20, 25, 30, 35, 40, 45]

do_eval = False

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

## ========================= ##
## ======= Evaluate  ======= ##
## ========================= ##

start = timer()
try:

    for run, out_size in zip(wRuns, outputs):
        for epoch in wEpchs:
            filename = Embed(run, epoch, post)
            try:
                images, pred, meta, labels, masks = pickle.load(open(filename, 'br'))
            except:
                from Network.data import load_nodule_dataset, load_nodule_raw_dataset, prepare_data
                from Network.model import miniXception_loader
                from Network.siameseArch import siamArch
                from Network.directArch import directArch

                # prepare test data
                images, labels, classes, masks, meta = \
                    prepare_data(load_nodule_dataset(size=size, res=res, sample=sample)[DataSubSet],
                                 categorize=False,
                                 reshuffle=False,
                                 return_meta=True,
                                 verbose=1)
                images = np.array([crop_center(im, msk, size=in_size)[0]
                                         for im, msk in zip(images, masks)])
                print("Image size changed to {}".format(images.shape))
                print('Mask not updated')
                if network == 'dir':
                    model = directArch(miniXception_loader, input_shape, objective="malignancy", output_size=out_size, normalize=normalize, pooling='max')
                elif network == 'siam':
                    model = siamArch(miniXception_loader, input_shape, distance='l2', output_size=out_size, normalize=normalize, pooling='rmac')
                elif network == 'dirR':
                    model = directArch(miniXception_loader, input_shape, objective="rating", output_size=out_size,
                                       normalize=normalize, pooling='max')
                elif network == 'siamR':
                    model = siamArch(miniXception_loader, input_shape, distance='l2', output_size=out_size, normalize=normalize, pooling='rmac', objective="rating")
                else:
                    assert(False)
                w = Weights(run=run, epoch=epoch)
                assert(w is not None)
                if network=='dir':
                    embed_model = model.extract_core(weights=w, repool=False)
                else:
                    embed_model = model.extract_core(weights=w)
                pred = embed_model.predict(images)
                pickle.dump((images, pred, meta, labels, masks), open(filename, 'bw'))
                #pickle.dump(((50*images).astype('int8'), (1000*np.abs(pred)).astype('uint8'), meta, labels, masks.astype('bool')),
                #           open(filename, 'bw'))
                print("Saved: {}".format(filename))

            print("Data: images({}), labels({})".format(images[0].shape, labels.shape))
            print("Range = [{:.2f},{:.2f}]".format(np.min(images[0]), np.max(images[0])))

            if do_eval:
                print(pred.shape)
                calc_embedding_statistics(pred, title=filename)

                plt.figure()
                #plt.subplot(211)
                plt.plot(np.transpose( np.squeeze(pred[np.argwhere(np.squeeze(labels==0)),
                                       np.squeeze(np.argwhere(np.std(pred, axis=0) > 0.0))])), 'blue', alpha=0.3)
                #plt.subplot(212)
                plt.plot(np.transpose( np.squeeze(pred[np.argwhere(np.squeeze(labels == 1)),
                                       np.squeeze(np.argwhere(np.std(pred, axis=0) > 0.0))])), 'red', alpha=0.2)


finally:
    plt.show()

    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))