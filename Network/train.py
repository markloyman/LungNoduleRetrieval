import gc
import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
np.random.seed(1337) # for reproducibility
import random
random.seed(1337)
from keras import backend as K
import tensorflow as tf
tf.set_random_seed(1234)
K.set_session(tf.Session(graph=tf.get_default_graph()))

from Network.DataGenSiam import DataGenerator
from Network.model import miniXception_loader
from Network.siameseArch import siamArch


import argparse
parser = argparse.ArgumentParser(description="Train Lung Nodule Retrieval NN")
group = parser.add_mutually_exclusive_group()
parser.add_argument("-e", "--epochs", type=int, help="epochs", default=0)
args = parser.parse_args()


## --------------------------------------- ##
## ------- General Setup ----------------- ##
## --------------------------------------- ##

data_size  = 144
model_size = 128
res    = 'Legacy' #'Legacy' #0.7 #0.5
sample = 'Normal' #'UniformNC' #'Normal' #'Uniform'
input_shape = (model_size, model_size, 1)

out_size = 128

epochs = args.epochs if (args.epochs != 0) else 20

# DIR / SIAM
choose_model = "SIAM"

## --------------------------------------- ##
## ------- Run Direct Architecture ------- ##
## --------------------------------------- ##

if choose_model is "DIR":
    run = 'dir000'

## --------------------------------------- ##
## ------- Run Siamese Architecture ------ ##
## --------------------------------------- ##

if choose_model is "SIAM":
    #run = 'siam049X' # data=128, regen dataset with seed
    #run = 'siam050'  # data=144, regen dataset with seed
    #run = 'siam051' # 144.Legacy.Normal, lr=e-3, l2 normalize, pooling=rmac, out=128, margin=1, data-aug {none}, sample-w(count)
    #run = 'siam052'  # 144.Legacy.Normal, lr=e-3, l2 normalize, pooling=rmac, out=128, margin=1, data-aug {none}, sample-w(dummy)
    #run = 'siam053'  # 144.Legacy.Normal, lr=e-3, l2 normalize, pooling=rmac, out=128, margin=1, data-aug {dummy}, sample-w(dummy)
    #run = 'siam054'  # 144.Legacy.Normal, lr=e-3, l2 normalize, pooling=rmac, out=128, margin=1, data-aug {'crop_stdev': 0.1}, sample-w(dummy)
    #run = 'siam055' # batch size is 128, data-aug {'crop_stdev': 0.1}, sample-w(dummy)
    #run = 'siam056'  # batch size is 64: (learning_rate=1e-3, decay=2e-5), data-aug {'crop_stdev': 0.1}, sample-w(dummy)
    #run = 'siam057'  # batch size is 64: (learning_rate=1e-2, decay=0.2), data-aug {'crop_stdev': 0.1}, sample-w(dummy)
    #run = 'siam058'  # 144.Legacy.Normal, lr=e-3, l2 normalize, pooling=rmac, out=128, margin=1, data-aug {none}, sample-w(none)
    #run = 'siam059X'  # 144.0.5.Normal, lr=e-3, l2 normalize, pooling=rmac, out=128, margin=1, data-aug {none}, sample-w(none)
    #run = 'siam060XX'  # 144.0.7.Normal, lr=e-3, l2 normalize, pooling=rmac, out=128, margin=1, data-aug {none}, sample-w(none)
    #run = 'siam061'  # val_factor=2, 144.Legacy.Normal, lr=e-3, l2 normalize, pooling=rmac, out=128, margin=1, data-aug {none}, sample-w(none)
    #run = 'siam062X'  # val_factor=2, 144.0.5.Normal, lr=e-3, l2 normalize, pooling=rmac, out=128, margin=1, data-aug {none}, sample-w(none)
    #run = 'siam063X'  # balanced sampling, val_factor=2, 144.Legacy.Normal, lr=e-3, l2 normalize, pooling=rmac, out=128, margin=1, data-aug {none}, sample-w(none)
    #run = 'siam063XX'  # balanced sampling, val_factor=2, 144.Legacy.Normal, lr=e-3, l2 normalize, pooling=rmac, out=128, margin=1, data-aug {e20}, sample-w(none)
    run = 'siam064'  # balanced sampling, val_factor=2, 144.Legacy.Normal, lr=e-3, l2 normalize, pooling=rmac, out=128, margin=1, data-aug {none}, sample-w(none)

    # model
    data_augment_params = {'max_angle': 0, 'flip_ratio': 0.0, 'crop_stdev': 0.1, 'epoch': 20}

    generator = DataGenerator(data_size=data_size, model_size=model_size, res=res, sample=sample, batch_sz=64,
                              val_factor=2, balanced=True,
                              do_augment=False, augment=data_augment_params,
                              use_class_weight=False)

    model = siamArch(miniXception_loader, input_shape, output_size=out_size,
                     distance='l2', normalize=True, pooling='rmac')
    model.model.summary()
    model.compile(learning_rate=1e-3, decay=0)

    model.load_generator(generator)

    model.train(label=run, n_epoch=epochs, gen=True)

    K.clear_session()
    gc.collect()