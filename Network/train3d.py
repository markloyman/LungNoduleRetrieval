import os
import numpy as np
import random
from keras import backend as K
import tensorflow as tf
# for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
random.seed(1337)
tf.set_random_seed(1234)
K.set_session(tf.Session(graph=tf.get_default_graph()))
from config import local, input_dir
from Network.Common.losses import pearson_correlation
from Network.Direct.directArch import DirectArch
from Network.Direct.DataGenDirect import DataGeneratorDir
from Network.Siamese.siameseArch import SiamArch
from Network.Siamese.DataGenSiam import DataGeneratorSiam
from Network.Triplet.tripletArch import TripArch
from Network.Triplet.DataGenTrip import DataGeneratorTrip
from Network.model3d import gru3d_loader
from Network.data_loader import load_nodule_dataset, build_loader_3d
from Network.Direct import prepare_data_direct
from Network.dataUtils import crop_center


def run(choose_model="DIR", epochs=200, config=0, skip_validation=False, no_training=False):

    np.random.seed(1337)
    random.seed(1337)
    tf.set_random_seed(1234)
    K.set_session(tf.Session(graph=tf.get_default_graph()))

    ## --------------------------------------- ##
    ## ------- General Setup ----------------- ##
    ## --------------------------------------- ##

    net_type = 'flat'  # 'flat', 'rmac'

    #data
    dataset_type = '3d'
    res = 0.5  # 'Legacy' #0.7 #0.5 #'0.5I'
    sample = 'Normal'  # 'UniformNC' #'Normal' #'Uniform'
    use_gen = True
    data_size = 160

    data_loader = build_loader_3d(configuration=config, net_type='dirR', run='251', epoch=60)

    # model
    out_size = 128
    if net_type == 'flat':
        model_size = 8*8*128
    elif net_type == 'rmac':
        model_size = 128
    else:
        assert False
    input_shape = (None, model_size)
    do_augment = False
    normalize = True

    print("-"*30)
    print("Running Sequence {} for --** {} **-- model, with #{} configuration".
          format("training" if not no_training else "validation", choose_model, config))
    print("\tdata_size = {},\n\tmodel_size = {},\n\tres = {},\n\tdo_augment = {}".
          format(data_size, model_size, res, do_augment))
    print("-" * 30)

    model = None

    ## --------------------------------------- ##
    ## ------- Prepare Direct Architecture ------- ##
    ## --------------------------------------- ##

    if choose_model is "DIR_RATING":

        #run = '0004'
        #run = '0005'  # new dataset (Train-Valid-Test)
        #run = '0006'  # ConvLSTM
        #run = '0007'  # ConvLSTM batch norm
        #run = '0008'  # ConvLSTM batch norm seperated activation (and relu instead of tangh)
        # run = '0008b'  # ConvLSTM batch norm seperated activation (tangh)
        # run = '0009'  # ConvLSTM no-batch-norm, binary
        # run = '0010'  # ConvLSTM no-batch-norm, binary-relu
        # run = '0011'  # ConvLSTM ~1M params (max pool)
        #run = '0012'  # ConvLSTM ~1M params (max pool) no relu
        #run = '0013'  # ConvLSTM ~1M params (max pool) batch norm is back
        run = '0014'  # ConvLSTM ~1M params (max pool) type2

        obj = 'rating'  # 'distance-matrix' 'rating' 'rating-size'


        rating_scale = 'none'
        reg_loss = None  # {'SampleCorrelation': 0.0}  # 'Dispersion', 'Std', 'FeatureCorrelation', 'SampleCorrelation'
        batch_size = 16

        epoch_pre = 20
        preload_weight = None  # FileManager.Weights('dirR', output_dir=input_dir).name(run='251c{}'.format(config), epoch=epoch_pre)

        model = DirectArch(gru3d_loader, input_shape, output_size=out_size, objective=obj, separated_prediction=False, binary=False,
                           normalize=normalize, pooling='max', l1_regularization=None, regularization_loss=reg_loss, batch_size=batch_size)
        model.model.summary()

        if preload_weight is not None:
            model.load_core_weights(preload_weight)

        # scheduale 02
        should_use_scheduale = (reg_loss is not None) or (obj == 'rating_size')
        sched = [{'epoch': 00, 'weights': [0.1, 0.9]},
                 {'epoch': 20, 'weights': [0.4, 0.6]},
                 {'epoch': 40, 'weights': [0.6, 0.4]},
                 {'epoch': 60, 'weights': [0.9, 0.1]},
                 {'epoch': 80, 'weights': [1.0, 0.0]}] \
            if should_use_scheduale else []

        loss = 'logcosh' if obj is not 'distance-matrix' else pearson_correlation
        model.compile(learning_rate=1e-3, decay=0, loss=loss, scheduale=sched, temporal_weights=False) # mean_squared_logarithmic_error, binary_crossentropy, logcosh

        if use_gen:
            generator = DataGeneratorDir(data_loader, val_factor=0 if skip_validation else 1,
                    data_size=data_size, model_size=model_size, batch_size=batch_size,
                    objective=obj, rating_scale=rating_scale, weighted_rating=False,
                    seq_model=True, balanced=False, do_augment=do_augment,
                    use_class_weight=False, use_confidence=False)
            model.load_generator(generator)
        else:
            dataset = load_nodule_dataset(size=data_size, res=res, sample=sample, dataset_type=dataset_type)
            images_train, labels_train, masks_train = prepare_data_direct(dataset[2], objective='rating', rating_scale=rating_scale)
            images_valid, labels_valid, masks_valid = prepare_data_direct(dataset[1], objective='rating', rating_scale=rating_scale)
            images_train = np.array([crop_center(im, msk, size=model_size)[0]
                               for im, msk in zip(images_train, masks_train)])
            images_valid = np.array([crop_center(im, msk, size=model_size)[0]
                               for im, msk in zip(images_valid, masks_valid)])
            model.load_data(images_train, labels_train, images_valid, labels_valid, batch_size=batch_size)

    ## --------------------------------------- ##
    ## ------- Prepare Siamese Architecture ------ ##
    ## --------------------------------------- ##

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
        #run = '100c'  # l2, max-pooling, train_factor=1
        #run = '101b'  # l2, max-pooling, train_factor=2
        #run = '102'  # l2, max-pooling, train_factor=3
        #run = '103'  # l2, max-pooling, train_factor=1, mse
        #run = '110'  # l2, max-pooling, train_factor=3
        #run = '112'  # l2, msrmac-pooling, train_factor=3
        #run = '122'  # l2, msrmac-pooling, train_factor=2, data-aug
        #run = '132'  # l2, msrmac-pooling, train_factor=2, data-aug, primary
        #run = '142'  # l2, msrmac-pooling, train_factor=2, out=64
        #run = '152'  # l2, msrmac-pooling, train_factor=2, out=32
        #run = '162'  # l2, msrmac-pooling, train_factor=2, out=8
        #run = '172'  # l2, msrmac-pooling, train_factor=2, out=256
        #run = '135'  # l2, msrmac-pooling, train_factor=2, data-aug, primary
        #run = '180'  # baseline, b64
        #run = '181'  # baseline, FeatCorr.1
        #run = '182'  # baseline, SampCorr.1

        run = '200'  # pretrain with dirR251-70

        #run = '300'   # obj: size
        #run = '311'  # obj: rating-size

        #run = 'zzz'  #

        dataset_type = 'Primary'
        obj = 'rating'  # rating / size / rating_size
        batch_size = 16 if local else 64
        reg_loss = None  # {'SampleCorrelation': 0.1}  # 'Dispersion', 'Std', 'FeatureCorrelation', 'SampleCorrelation'

        epoch_pre = 60
        preload_weight = None  # FileManager.Weights('dirR', output_dir=input_dir).name(run='251c{}'.format(config), epoch=70)

        should_use_scheduale = (reg_loss is not None) or (obj == 'rating_size')
        '''
        sched = [{'epoch': 00, 'weights': [0.1, 0.9]},
                 {'epoch': 30, 'weights': [0.4, 0.6]},
                 {'epoch': 60, 'weights': [0.6, 0.4]},
                 {'epoch': 80, 'weights': [0.9, 0.1]},
                 {'epoch': 100, 'weights': [1.0, 0.0]}] \
            if should_use_scheduale else []
        '''
        sched = [{'epoch': 00, 'weights': [0.1, 0.9]},
                 {'epoch': 20, 'weights': [0.4, 0.6]},
                 {'epoch': 30, 'weights': [0.6, 0.4]},
                 {'epoch': 50, 'weights': [0.9, 0.1]},
                 {'epoch': 80, 'weights': [1.0, 0.0]}] \
            if should_use_scheduale else []
        # model
        generator = DataGeneratorSiam(data_loader,
                                      data_size=data_size, model_size=model_size, batch_size=batch_size,
                                      train_facotr=2, val_factor=0 if skip_validation else 3, balanced=False,
                                      objective=obj,
                                      do_augment=do_augment, augment=data_augment_params,
                                      use_class_weight=False, use_confidence=False)

        model = SiamArch(miniXception_loader, input_shape, output_size=out_size, objective=obj,
                         batch_size=batch_size, distance='l2', normalize=normalize, pooling='msrmac',
                         regularization_loss=reg_loss, l1_regularization=False)

        if preload_weight is not None:
            model.load_core_weights(preload_weight)
        model.model.summary()
        model.compile(learning_rate=1e-3, decay=0, loss='logcosh', scheduale=sched) # mean_squared_error, logcosh
        model.load_generator(generator)

    ## --------------------------------------- ##
    ## ------- Prepare Triplet Architecture ------ ##
    ## --------------------------------------- ##

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
        #run = 'trip028'  # obj:malig, max, categorize, no-decay

        run = 'trip_100'  # obj:malig, msrmac, softplus-loss
        #run = 'trip101'  # obj:malig, msrmac, rank-loss

        dataset_type = 'Primary'
        objective = 'malignancy'
        use_rank_loss = False

        gen = True
        preload_weight = None #'./Weights/w_dirR011X_50.h5'

        # model
        model = TripArch(miniXception_loader, input_shape, objective=objective, output_size=out_size,
                         distance='l2', normalize=True, pooling='msrmac', categorize=use_rank_loss)

        if preload_weight is not None:
            model.load_core_weights(preload_weight)
        model.model.summary()
        model.compile(learning_rate=1e-3, decay=0) #0.05

        generator = DataGeneratorTrip(data_loader,
                                      data_size=data_size, model_size=model_size,
                                      batch_size=16,
                                      objective=objective, balanced=(objective == 'malignancy'), categorize=True,
                                      val_factor=0 if skip_validation else 3, train_factor=1,
                                      do_augment=do_augment, augment=data_augment_params,
                                      use_class_weight=False, use_confidence=False)
        if gen:
            model.load_generator(generator)
        else:
            imgs_trn, lbl_trn = generator.next_train().__next__()
            imgs_val, lbl_val = generator.next_val().__next__()
            model.load_data(imgs_trn, lbl_trn,imgs_val, lbl_val)

    ## --------------------------------------- ##
    ## -------      RUN             ------ ##
    ## --------------------------------------- ##

    print('Current Run: {}{}c{}'.format('', run, config))

    if no_training:
        model.last_epoch = epochs
        model.run='{}{}c{}'.format('', run, config)
    else:
        model.train(run='{}{}c{}'.format('', run, config), epoch=(0 if preload_weight is None else epoch_pre), n_epoch=epochs, gen=use_gen, do_graph=False)

    return model
