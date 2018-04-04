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
    from Network.Direct.directArch import directArch
    from Network.Direct.DataGenDirect import DataGeneratorDir
    from Network.Siamese.siameseArch import siamArch
    from Network.Siamese.DataGenSiam import DataGenerator
    from Network.Triplet.tripletArch import tripArch
    from Network.Triplet.DataGenTrip import DataGeneratorTrip
    from Network.model import miniXception_loader
    from Network.data_loader import load_nodule_dataset, prepare_data_direct
    from Network.dataUtils import crop_center
except:
    # Paths for floyd cloud
    from Direct.directArch import directArch
    from Direct.DataGenDirect import DataGeneratorDir
    from Siamese.siameseArch import siamArch
    from Siamese.DataGenSiam import DataGenerator
    from Triplet.tripletArch import tripArch
    from Triplet.DataGenTrip import DataGeneratorTrip
    from model import miniXception_loader
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


# DIR / SIAM / DIR_RATING / SIAM_RATING
def run(choose_model = "DIR"):

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

    epochs = args.epochs if (args.epochs != 0) else 200


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
            images_train, labels_train, class_train, masks_train, _ = prepare_data_direct(dataset[2], classes=2, size=model_size)
            images_valid, labels_valid, class_valid, masks_valid, _ = prepare_data_direct(dataset[1], classes=2, size=model_size)
            images_train = np.array([crop_center(im, msk, size=model_size)[0]
                               for im, msk in zip(images_train, masks_train)])
            images_valid = np.array([crop_center(im, msk, size=model_size)[0]
                               for im, msk in zip(images_valid, masks_valid)])
            model.load_data(images_train, labels_train, images_valid, labels_valid, batch_sz=32)

        model.train(label=run, n_epoch=epochs, gen=use_gen, do_graph=False)

    if choose_model is "DIR_RATING":
        #run = 'dirR000'  # mse
        #run = 'dirR001'  # msle
        #run = 'dirR002'  # mse, linear activation
        #run = 'dirR003'  # mse, linear activation
        #run = 'dirR004X'  # logcosh
        #run = 'dirR005'  # binary_crossentropy
        #run = 'dirR006'  # mse, batch_sz=64
        #run = 'dirR007'  # mse, data-aug
        #run = 'dirR008'  # mse, scaled-rating
        #run = 'dirR009'  # logcosh, max, scaled
        #run = 'dirR010'  # logcosh, rmac, scaled
        #run = 'dirR011X'  # logcosh, rmac, no scaling
        #run = 'dirR012'  # logcosh, rmac, no scaling, decay
        #run = 'dirR013'  # model-binary, rmac, logcosh
        run = 'dirR014ZZ'  # model-not-binary, rmac, logcosh

        rating_scale = 'none'
        use_gen = True
        obj = 'rating'

        model = directArch(miniXception_loader, input_shape, output_size=out_size, objective=obj,
                           normalize=normalize, pooling='rmac', binary=False)
        model.model.summary()
        model.compile(learning_rate=1e-3, decay=1e-3, loss='logcosh') # mean_squared_logarithmic_error, binary_crossentropy, logcosh

        if use_gen:
            data_augment_params = {'max_angle': 0, 'flip_ratio': 0.1, 'crop_stdev': 0.05, 'epoch': 0}
            generator = DataGeneratorDir(
                    data_size=data_size, model_size=model_size, res=res, sample=sample, batch_sz=32,
                    objective=obj, categorize=False, rating_scale=rating_scale,
                    val_factor=1, balanced=False,
                    do_augment=False, augment=data_augment_params,
                    use_class_weight=False, class_weight='balanced')
            model.load_generator(generator)
        else:
            dataset = load_nodule_dataset(size=data_size, res=res, sample=sample)
            images_train, labels_train, masks_train = prepare_data_direct(dataset[2], size=model_size, objective='rating', rating_scale=rating_scale)
            images_valid, labels_valid, masks_valid = prepare_data_direct(dataset[1], size=model_size, objective='rating', rating_scale=rating_scale)
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
        gen = True
        # model
        data_augment_params = {'max_angle': 0, 'flip_ratio': 0.1, 'crop_stdev': 0.05, 'epoch': 0}
        generator = DataGenerator(data_size=data_size, model_size=model_size, res=res, sample=sample, batch_sz=64,
                                  val_factor=1, balanced=True, objective="malignancy",
                                  do_augment=False, augment=data_augment_params,
                                  use_class_weight=False)

        model = siamArch(miniXception_loader, input_shape, output_size=out_size,
                         distance='l2', normalize=normalize, pooling='max')
        model.model.summary()
        model.compile(learning_rate=1e-3, decay=0)
        if gen:
            model.load_generator(generator)
        else:
            imgs_trn, lbl_trn = generator.next_train().__next__()
            imgs_val, lbl_val = generator.next_val().__next__()
            model.load_data(imgs_trn, lbl_trn, imgs_val, lbl_val)

        model.train(label=run, n_epoch=epochs, gen=gen)

    if choose_model is "SIAM_RATING":

        #run = 'siamR001'  # mse-loss, rating-scaled
        #run = 'siamR002'  # mse-loss, rating-scaled, repeated-epochs
        #run = 'siamR003'  # mse-loss, 0.25*rating-scaled, repeated-epochs(3)
        #run = 'siamR004X'  # mse-loss, 0.25*rating-scaled, repeated-epochs(5)
        #run = 'siamR005'  # mse-loss, 0.25*rating-scaled, repeated-epochs(1)
        #run = 'siamR006XX'  # rmac, mse-loss, 0.25*rating-scaled, repeated-epochs(1)
        #run = 'siamR007'  # rmac, logcosh-loss, 0.25*rating-scaled, repeated-epochs(1)
        #run = 'siamR008X'  # data-aug
        #run = 'siamR009'  # cosine

        # model
        data_augment_params = {'max_angle': 0, 'flip_ratio': 0.1, 'crop_stdev': 0.05, 'epoch': 0}
        generator = DataGenerator(data_size=data_size, model_size=model_size, res=res, sample=sample, batch_sz=32,
                                  val_factor=3, balanced=False,
                                  objective="rating",
                                  do_augment=False, augment=data_augment_params,
                                  use_class_weight=False)

        model = siamArch(miniXception_loader, input_shape, output_size=out_size, objective="rating",
                         distance='l2', normalize=normalize, pooling='rmac')
        model.model.summary()
        model.compile(learning_rate=1e-3, decay=0, loss='mean_squared_error') # mean_squared_error, logcosh
        model.load_generator(generator)

        model.train(label=run, n_epoch=epochs, gen=True)

    if choose_model is "TRIPLET":

        #run = 'trip011XXX'  # mrg-loss, decay(0.01), max-pool
        #run = 'trip012X'  # mrg-loss, decay(0.05), rmac-pool
        #run = 'trip013'  # cosine
        #run = 'trip014' # ortogonal initialization
        #run = 'trip015X'  # objective rating
        #run = 'trip016XXXX'  # softplus-loss
        #run = 'trip017'  # softplus-loss, no decay
        #run = 'trip018'  # binary
        #run = 'trip019'  # categorize
        #run = 'trip020X'  # rating-conf-tryout

        #run = 'trip021' # pretrained
        #run = 'trip022XXX'  # pretrained rmac
        #run = 'trip023X'  # pretrained categorize
        #run = 'trip024'  # pretrained confidence
        #run = 'trip025'  # pretrained cat,conf
        #run = 'trip026Z'  # class_weight='rating_distance', cat

        #run = 'trip027'  # obj:malig, rmac, categorize, no-decay
        run = 'trip028'  # obj:malig, max, categorize, no-decay

        gen = True
        preload_weight = None #'./Weights/w_dirR011X_50.h5'


        # model
        model = tripArch(miniXception_loader, input_shape, output_size=out_size,
                         distance='l2', normalize=True, pooling='max', categorize=True, binary=False)
        if preload_weight is not None:
            model.load_core_weights(preload_weight)
        model.model.summary()
        model.compile(learning_rate=1e-3, decay=0) #0.05
        data_augment_params = {'max_angle': 0, 'flip_ratio': 0.1, 'crop_stdev': 0.05, 'epoch': 0}
        generator = DataGeneratorTrip(data_size=data_size, model_size=model_size, res=res, sample=sample, batch_sz=80,
                                      val_factor=3, objective="malignancy",
                                      do_augment=False, augment=data_augment_params,
                                      use_class_weight=False, class_weight='rating_distance')
        if gen:
            model.load_generator(generator)
        else:
            imgs_trn, lbl_trn = generator.next_train().__next__()
            imgs_val, lbl_val = generator.next_val().__next__()
            model.load_data(imgs_trn, lbl_trn,imgs_val, lbl_val)

        model.train(label=run, n_epoch=epochs, gen=gen)

    K.clear_session()
    gc.collect()


if __name__ == "__main__":
    # DIR / SIAM / DIR_RATING / SIAM_RATING / TRIPLET
    run('TRIPLET')