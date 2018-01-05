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
try:
    from Network.DataGenSiam import DataGenerator
    from Network.DataGenDirect import DataGeneratorDir
    from Network.model import miniXception_loader
    from Network.siameseArch import siamArch
    from Network.directArch import directArch
    from Network.data import load_nodule_dataset, prepare_data_direct
    from Network.dataUtils import crop_center
except:
    from DataGenSiam import DataGenerator
    from DataGenDirect import DataGeneratorDir
    from model import miniXception_loader
    from siameseArch import siamArch
    from directArch import directArch
    from data import load_nodule_dataset, prepare_data_direct
    from dataUtils import crop_center

    import os, errno

    try:
        os.makedirs('/output/Weights/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.makedirs('/output/logs/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.makedirs('/output/history/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

import argparse
parser = argparse.ArgumentParser(description="Train Lung Nodule Retrieval NN")
parser.add_argument("-e", "--epochs", type=int, help="epochs", default=0)
args = parser.parse_args()


## --------------------------------------- ##
## ------- General Setup ----------------- ##
## --------------------------------------- ##

#data
data_size  = 144
res    = '0.5I' #'Legacy' #0.7 #0.5
sample = 'Normal' #'UniformNC' #'Normal' #'Uniform'
#model
model_size = 128
input_shape = (model_size, model_size, 1)
normalize = True
out_size = 128

epochs = args.epochs if (args.epochs != 0) else 60

# DIR / SIAM / DIR_RATING / SIAM_RATING
choose_model = "SIAM_RATING"

print("Running training for --** {} **-- model".format(choose_model))

## --------------------------------------- ##
## ------- Run Direct Architecture ------- ##
## --------------------------------------- ##

if choose_model is "DIR":
    #run = 'dir100'  # no-pooling
    #run = 'dir101'  # conv-pooling
    #run = 'dir102'  # avg-pooling
    #run = 'dir103'  # max-pooling
    #run = 'dir104'  # rmac-pooling
    run = 'dir_'  #

    use_gen = True

    model = directArch( miniXception_loader, input_shape, output_size=out_size,
                        normalize=normalize, pooling='rmac')
    model.model.summary()
    model.compile(learning_rate=1e-3, decay=1e-5)
    if use_gen:
        data_augment_params = {'max_angle': 0, 'flip_ratio': 0.1, 'crop_stdev': 0.05, 'epoch': 0}
        generator = DataGeneratorDir(
                        data_size=data_size, model_size=model_size, res=res, sample=sample, batch_sz=32,
                        val_factor=1, balanced=False,
                        do_augment=False, augment=data_augment_params,
                        use_class_weight=True, class_weight='balanced')
        model.load_generator(generator)
    else:
        dataset = load_nodule_dataset(size=data_size, res=res, sample=sample)
        images_train, labels_train, masks_train = prepare_data_direct(dataset[2], classes=2, size=model_size)
        images_valid, labels_valid, masks_valid = prepare_data_direct(dataset[1], classes=2, size=model_size)
        images_train = np.array([crop_center(im, msk, size=model_size)[0]
                           for im, msk in zip(images_train, masks_train)])
        images_valid = np.array([crop_center(im, msk, size=model_size)[0]
                           for im, msk in zip(images_valid, masks_valid)])
        model.load_data(images_train, labels_train, images_valid, labels_valid, batch_sz=32)

    model.train(label=run, n_epoch=epochs, gen=use_gen)

if choose_model is "DIR_RATING":
    #run = 'dirR000'  # mse
    #run = 'dirR001'  # msle
    #run = 'dirR002'  # mse, linear activation
    #run = 'dirR003'  # mse, linear activation
    #run = 'dirR004'  # logcosh
    #run = 'dirR005'  # binary_crossentropy
    #run = 'dirR006'  # mse, batch_sz=64
    run = 'dirR007'  # mse, data-aug

    use_gen = True

    model = directArch(miniXception_loader, input_shape, output_size=out_size, objective='rating',
                       normalize=normalize, pooling='max')
    model.model.summary()
    model.compile(learning_rate=1e-3, decay=1e-5, loss='mean_squared_error') # mean_squared_logarithmic_error, binary_crossentropy, logcosh

    if use_gen:
        data_augment_params = {'max_angle': 0, 'flip_ratio': 0.1, 'crop_stdev': 0.05, 'epoch': 0}
        generator = DataGeneratorDir(
                data_size=data_size, model_size=model_size, res=res, sample=sample, batch_sz=32,
                objective='rating',
                val_factor=1, balanced=False,
                do_augment=True, augment=data_augment_params,
                use_class_weight=False, class_weight='balanced')
        model.load_generator(generator)
    else:
        dataset = load_nodule_dataset(size=data_size, res=res, sample=sample)
        images_train, labels_train, masks_train = prepare_data_direct(dataset[2], size=model_size, objective='rating')
        images_valid, labels_valid, masks_valid = prepare_data_direct(dataset[1], size=model_size, objective='rating')
        images_train = np.array([crop_center(im, msk, size=model_size)[0]
                           for im, msk in zip(images_train, masks_train)])
        images_valid = np.array([crop_center(im, msk, size=model_size)[0]
                           for im, msk in zip(images_valid, masks_valid)])
        model.load_data(images_train, labels_train, images_valid, labels_valid, batch_sz=32)

    model.train(label=run, n_epoch=epochs, gen=use_gen, do_graph=False)

## --------------------------------------- ##
## ------- Run Siamese Architecture ------ ##
## --------------------------------------- ##

if choose_model is "SIAM":

    #run = 'siam100'  # base model, res=0.5I, l2
    #run = 'siam101'  # base model, res=0.5I, l1 | b: lr->e-2
    #run = 'siam102'  # base model, res=0.5I, l2, msrmac
    #run = 'siam103'  # base model, res=0.5I, cosine
    #run = 'siam104'  # base model, res=0.5I, l1, norm-l1
    run = 'siam999'  # junk

    # model
    data_augment_params = {'max_angle': 0, 'flip_ratio': 0.1, 'crop_stdev': 0.05, 'epoch': 0}
    generator = DataGenerator(data_size=data_size, model_size=model_size, res=res, sample=sample, batch_sz=64,
                              val_factor=5, balanced=True,
                              do_augment=False, augment=data_augment_params,
                              use_class_weight=False)

    model = siamArch(miniXception_loader, input_shape, output_size=out_size,
                     distance='l2', normalize=normalize, pooling='rmac')
    model.model.summary()
    model.compile(learning_rate=1e-3, decay=0)
    model.load_generator(generator)

    model.train(label=run, n_epoch=epochs, gen=True)

if choose_model is "SIAM_RATING":

    run = 'siamR999'  #

    # model
    data_augment_params = {'max_angle': 0, 'flip_ratio': 0.1, 'crop_stdev': 0.05, 'epoch': 0}
    generator = DataGenerator(data_size=data_size, model_size=model_size, res=res, sample=sample, batch_sz=64,
                              val_factor=1, balanced=True,
                              objective="rating",
                              do_augment=False, augment=data_augment_params,
                              use_class_weight=False)

    model = siamArch(miniXception_loader, input_shape, output_size=out_size, objective="rating",
                     distance='l2', normalize=normalize, pooling='rmac')
    model.model.summary()
    model.compile(learning_rate=1e-3, decay=0)
    model.load_generator(generator)

    model.train(label=run, n_epoch=epochs, gen=True)

K.clear_session()
gc.collect()