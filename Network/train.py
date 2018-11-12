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
try:
    from Network.Common.losses import pearson_correlation
    from Network.Direct.directArch import DirectArch
    from Network.Direct.DataGenDirect import DataGeneratorDir
    from Network.Siamese.siameseArch import SiamArch
    from Network.Siamese.DataGenSiam import DataGeneratorSiam
    from Network.Triplet.tripletArch import TripArch
    from Network.Triplet.DataGenTrip import DataGeneratorTrip
    from Network.model import miniXception_loader
    from Network.data_loader import load_nodule_dataset, build_loader
    from Network.Direct import prepare_data_direct
    from Network.dataUtils import crop_center
    from Network import FileManager
    input_dir = './output'
    local = True
except:
    # Paths for floyd cloud
    from Common.losses import pearson_correlation
    from Direct.directArch import DirectArch
    from Direct.DataGenDirect import DataGeneratorDir
    from Siamese.siameseArch import SiamArch
    from Siamese.DataGenSiam import DataGeneratorSiam
    from Triplet.tripletArch import TripArch
    from Triplet.DataGenTrip import DataGeneratorTrip
    from model import miniXception_loader
    from data_loader import load_nodule_dataset, build_loader
    from Direct import prepare_data_direct
    from dataUtils import crop_center
    import FileManager
    input_dir = '/input'
    local = False

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
    try:
        os.makedirs('/output/embed/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# DIR / SIAM / DIR_RATING / SIAM_RATING


def run(choose_model="DIR", epochs=200, config=0, skip_validation=False, no_training=False):

    np.random.seed(1337)
    random.seed(1337)
    tf.set_random_seed(1234)
    K.set_session(tf.Session(graph=tf.get_default_graph()))

    ## --------------------------------------- ##
    ## ------- General Setup ----------------- ##
    ## --------------------------------------- ##

    #data
    dataset_type = 'Primary'
    data_size = 160
    if no_training:
        data_size = 160
    res = 0.5  # 'Legacy' #0.7 #0.5 #'0.5I'
    sample = 'Normal'  # 'UniformNC' #'Normal' #'Uniform'
    use_gen = True
    #model
    model_size = 128
    input_shape = (model_size, model_size, 1)
    normalize = True
    out_size = 128
    do_augment = True
    if no_training:
        do_augment = False
    preload_weight = None

    print("-"*30)
    print("Running {} for --** {} **-- model, with #{} configuration".
          format("training" if not no_training else "validation", choose_model, config))
    print("\tdata_size = {},\n\tmodel_size = {},\n\tres = {},\n\tdo_augment = {}".
          format(data_size, model_size, res, do_augment))
    print("\tdataset_type = {}".format(dataset_type))
    print("-" * 30)

    model = None

    data_augment_params = {'max_angle': 30, 'flip_ratio': 0.5, 'crop_stdev': 0.15, 'epoch': 0}

    data_loader = build_loader(size=data_size, res=res, sample=sample, dataset_type=dataset_type, configuration=config)

    ## --------------------------------------- ##
    ## ------- Prepare Direct Architecture ------- ##
    ## --------------------------------------- ##

    if choose_model is "DIR":
        # run = '300'  # SPIE avg-pool (data-aug, balanced=False,class_weight=True)
        # run = '301'  # SPIE max-pool (data-aug, balanced=False,class_weight=True)
        # run = '302'  # SPIE rmac-pool (data-aug, balanced=False,class_weight=True)

        # run = 'zzz'

        model = DirectArch( miniXception_loader, input_shape, output_size=out_size,
                            normalize=normalize, pooling='msrmac')
        model.model.summary()
        model.compile(learning_rate=1e-3, decay=0)
        if use_gen:
            generator = DataGeneratorDir(data_loader,
                            val_factor=0 if skip_validation else 1, balanced=False,
                            data_size=data_size, model_size=model_size, batch_size=32,
                            do_augment=do_augment, augment=data_augment_params,
                            use_class_weight=True, use_confidence=False)
            model.load_generator(generator)
        else:
            dataset = load_nodule_dataset(size=data_size, res=res, sample=sample)
            images_train, labels_train, class_train, masks_train, _ = prepare_data_direct(dataset[2], num_of_classes=2)
            images_valid, labels_valid, class_valid, masks_valid, _ = prepare_data_direct(dataset[1], num_of_classes=2)
            images_train = np.array([crop_center(im, msk, size=model_size)[0]
                               for im, msk in zip(images_train, masks_train)])
            images_valid = np.array([crop_center(im, msk, size=model_size)[0]
                               for im, msk in zip(images_valid, masks_valid)])
            model.load_data(images_train, labels_train, images_valid, labels_valid, batch_size=32)

    if choose_model is "DIR_RATING":

        ### CLEAN SET
        # run = '800'  # rmac conf:size
        # run = '801'  # rmac conf:none
        # run = '802'  # rmac conf:rating-std
        # run = '803'  # max conf:none

        ### PRIMARY SET
        # run = '810'  # rmac conf:size
        # run = '811'  # rmac conf:none
        # run = '812'  # rmac conf:rating-std
        run = '813'  # max conf:none

        # run = 'zzz'

        obj = 'rating'  # 'distance-matrix' 'rating' 'rating-size'

        rating_scale = 'none'
        reg_loss = None  # {'SampleCorrelation': 0.0}  # 'Dispersion', 'Std', 'FeatureCorrelation', 'SampleCorrelation'
        batch_size = 32

        epoch_pre = 20
        preload_weight = None  # FileManager.Weights('dirR', output_dir=input_dir).name(run='251c{}'.format(config), epoch=epoch_pre)

        model = DirectArch(miniXception_loader, input_shape, output_size=out_size, objective=obj, separated_prediction=False,
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
        model.compile(learning_rate=1e-3, decay=0, loss=loss, scheduale=sched)  # mean_squared_logarithmic_error, binary_crossentropy, logcosh

        if use_gen:
            generator = DataGeneratorDir(data_loader,
                    val_factor=0 if skip_validation else 1,
                    data_size=data_size, model_size=model_size, batch_size=batch_size,
                    objective=obj, rating_scale=rating_scale, weighted_rating=(obj=='distance-matrix'),
                    balanced=False,
                    do_augment=do_augment, augment=data_augment_params,
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

    if choose_model is "SIAM":
        # run = '300'  # l1, avg-pool (data-aug, balanced=True, class_weight=False)
        # run = '301'  # l1, max-pool (data-aug, balanced=True, class_weight=False)
        # run = '302'  # l1, rmac-pool (data-aug, balanced=True, class_weight=False)
        # run = '310'  # l2, avg-pool (data-aug, balanced=True, class_weight=False)
        # run = '311'  # l2, max-pool (data-aug, balanced=True, class_weight=False)
        # run = '312'  # l2, rmac-pool (data-aug, balanced=True, class_weight=False)
        # run = '320'  # cos, avg-pool (data-aug, balanced=True, class_weight=False)
        # run = '321'  # cos, max-pool (data-aug, balanced=True, class_weight=False)
        # run = '322b'  # cos, rmac-pool (data-aug, balanced=True, class_weight=False)

        # b/c - changed margin-loss params
        # run = '313c'  # l2, max-pool MARGINAL-LOSS (data-aug, balanced=True, class_weight=False)
        # run = '314c'  # l2, rmac-pool MARGINAL-LOSS (data-aug, balanced=True, class_weight=False)
        # run = '323c'  # cos, max-pool MARGINAL-LOSS (data-aug, balanced=True, class_weight=False)
        # run = '324c'  # cos, rmac-pool MARGINAL-LOSS (data-aug, balanced=True, class_weight=False)

        # run = 'zzz'

        batch_size = 64 if local else 128

        # model
        generator = DataGeneratorSiam(data_loader,
                                      data_size=data_size, model_size=model_size, batch_size=batch_size,
                                      val_factor=0 if skip_validation else 3, balanced=True, objective="malignancy",
                                      do_augment=do_augment, augment=data_augment_params,
                                      use_class_weight=False)

        model = SiamArch(miniXception_loader, input_shape, output_size=out_size, batch_size=batch_size,
                         distance='l2', normalize=normalize, pooling='msrmac')
        model.model.summary()
        model.compile(learning_rate=1e-3, decay=0)
        if use_gen:
            model.load_generator(generator)
        else:
            imgs_trn, lbl_trn = generator.next_train().__next__()
            imgs_val, lbl_val = generator.next_val().__next__()
            model.load_data(imgs_trn, lbl_trn, imgs_val, lbl_val)

    if choose_model is "SIAM_RATING":
        ### clean set
        # run = '400'  # l2-rmac no-conf
        # run = '401'  # cosine-rmac no-conf
        # run = '402'  # l2-rmac conf
        # run = '403'  # cosine-rmac conf
        # run = '404'  # l2-max no-conf
        # run = '405'  # cosine-max no-conf

        ### primary set
        # run = '410'  # l2-rmac no-conf
        # run = '411'  # cosine-rmac no-conf
        # run = '412'  # l2-rmac conf
        # run = '413'  # cosine-rmac conf
        # run = '414'  # l2-max no-conf
        # run = '415'  # cosine-max no-conf

        # run = 'zzz'

        obj = 'rating'  # rating / size / rating_size
        batch_size = 16 if local else 64
        reg_loss = None  # {'SampleCorrating_clusters_distance_and_stdrelation': 0.1}  # 'Dispersion', 'Std', 'FeatureCorrelation', 'SampleCorrelation'

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
                                      objective=obj, weighted_rating=True,
                                      do_augment=do_augment, augment=data_augment_params,
                                      use_class_weight=False, use_confidence=False)

        model = SiamArch(miniXception_loader, input_shape, output_size=out_size, objective=obj,
                         batch_size=batch_size, distance='cosine', normalize=normalize, pooling='rmac',
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
                                      data_size=data_size, model_size=model_size, batch_size=16,
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
        model.run = '{}{}c{}'.format('', run, config)
    else:
        model.train(run='{}{}c{}'.format('', run, config), epoch=(0 if preload_weight is None else epoch_pre), n_epoch=epochs, gen=use_gen, do_graph=False)

    return model
