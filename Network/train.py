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
    dataset_type = 'Clean'
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
        #run = 'dir100'  # no-pooling
        #run = 'dir101'  # conv-pooling
        #run = 'dir102'  # avg-pooling
        #run = 'dir103'  # max-pooling
        #run = 'dir104c'  # rmac-pooling
        #run = 'dir200'  # rmac, decay=0
        #run = 'dir201'  # max, decay=0
        #run = 'dir999'  # rmac
        #run  = 'dir900'   # max
        #run = 'dir901'  # avg
        #run = 'dir902'  # msrmac
        #run = 'dir210'  # out64
        #run = 'dir211'  # out256
        #run = 'dir212'  # out32
        #run = 'dir220'  # network max width 256
        #run = 'dir221'  # network max width 256, data aug ('crop_stdev': 0.05) - used 128 data
        #run = 'dir222'  # network max width 256, data aug ('crop_stdev': 0.15) - used 128 data
        #run = 'dir223'  # Sequence, 3 workers, data aug with 144 data
        #run = 'dir224'  # Sequence, 5 deg rotation, data aug with 144 data
        #run = 'dir225'  # Sequence, 0 deg rotation, data aug with 144 data, drop.5
        #run = 'dir226'  # Sequence, 0 deg rotation, data aug with 144 data, drop.3
        #run = 'dir227'  # Sequence, 0 deg rotation, data aug with 144 data, drop.0
        #run = 'dir228'  # Sequence, 0 deg rotation, data aug with 144 data, drop.1
        #run = 'dir229'  # 10 deg rotation, drop.1
        #run = 'dir230'  # 20 deg rotation, drop.1
        #run = 'dir231'  # 30 deg rotation, drop.1
        #run = 'dir232'  # no rotation, drop.1
        #run = 'dir233'  # no rotation, drop.1
        #run = 'dir234'  # 30 deg rotation, drop.1
        #run = '235'  # baseline with Dataset160
        #run = 'dir236'  # baseline with Dataset128
        #run = 'dir240b'  # data-aug-no-rot
        #run = 'dir241'  # data-aug-rot20
        #run = 'dir242'  # data-aug-rot40
        #run = 'dir243b'  # data-aug-rot60
        #run = 'dir250'  # max2-pooling
        #run = '251'  # avg-pooling
        #run = '252'  # msrmac-pooling
        #run = 'dir253'  # avg-pooling, aug
        #run = '254'  # msrmac-pooling, aug
        run = 'zzz'

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
        #run = 'dirR014ZZ'  # model-not-binary, rmac, logcosh

        #run = '201b'  # msrmac-pool
        #run = '202c'  # avg-pool

        #run = '200b'  # max-pool
        #run = '210'  # max-pool data-aug                                               Class Weight -> B:0.74, M:1.53
        #run = '220'  # max-pool data-aug data-primary                                  Class Weight -> B:0.77, M:1.52, U:??? (¬0.9)

        #run = '203'  # max-pool                            (rating w_mean)
        #run = '213'  # max-pool data-aug
        #run = '223'  # max-pool data-aug data-primary                                  Class Weight -> B:0.78, M:1.61, U:0.91


        #run = '233'  # max-pool data-aug data-primary, confidence                      Class Weight -> B:0.77, M: 1.52, U: 0.38
        #run = '243'  # max-pool data-aug data-primary, confidence no-class-weight
        #run = '253'  # max-pool data-aug data-primary, no-confidence no-class-weight

        #run = '251'  # msrmac-pool data-aug data-primary, no-confidence no-class-weight

        #run = '260'  # msrmac-pool no reg
        #run = '261'  # msrmac-pool L1 reg = 1e-2
        #run = '262'  # msrmac-pool L1 reg = 1e-3
        #run = '263'   # msrmac-pool L1 reg = 1e-3 (on embedding)
        #run = 'z264'  # msrmac-pool, RegLoss:FeatureCorrelation

        #run = '270'  # msrmac, baseline, b=32
        #run = '271'  # msrmac, baseline, b=128
        #run = '272'  # msrmac, b32, SampCorr.1
        #run = '273'  # msrmac, b32, SampCorr.5
        #run = '275'  # msrmac, b32, FeatCorr.1
        #run = '276'  # msrmac, FeatCorr.sch01
        #run = '277'  # msrmac, SampCorr.sch01

        #run = '300'  # msrmac, separated-predictions, data-aug, data-primary

        #run = '400b'  # obj:size
        #run = '401'  # obj:size, double dense prediction layer
        #run = '402z'  # obj:size, double dense prediction layer + BN
        #run = '410'  # obj:size(mask area), double dense prediction layer + BN
        #run = '411d'  # obj:size (digitized mask area)
        #run = '412'  # obj:size (digitized mask area) *corrected data-loading

        #run = '500'  # obj:rating_size
        #run = '501'  # obj:rating_size (digitized mask area)
        #run = '502' # schd1 *corrected data-loading
        #run = '503' # schd2 *corrected data-loading

        #run = '512c'  # primary aug

        #run = '600'  # base full aug
        #run = '601b'  # full aug + confidence
        #run = '602'  # full aug + confidence + b64
        #run = '603'  # full aug + confidence + b32 + lr-4
        #run = '604'  # full aug + confidence + pretrain:dirR251
        #run = '605'  # full aug + confidence + pretrain:dirR251 + lr-4
        #run = '606'

        #run = '700'  # distance-matrix-loss primary aug
        #run = '701'  # distance-matrix-loss primary aug (-corretion loss)
        #run = '702'  # distance-matrix-loss primary aug (-corretion loss) b64
        #run = '703'  # distance-matrix-loss primary aug (-corretion loss) b128
        #run = '704'  # distance-matrix, pretrain:dirR251
        #run = '705'  # distance-matrix, l2-correlation
        #run = '706'  # distance-matrix, l2-correlation, pretrain:dirR251
        #run = '707'  # distance-matrix, l2-correlation, pretrain:dirR251, lr-4, dm-labels
        #run = '708'  # distance-matrix, l2-correlation, pretrain:dirR251, lr-4, b64, dm-labels
        #run = '709'  # distance-matrix, l2-corr, dm-labels

        #run = '710'  # distance-matrix, l2-corr, dm-labels, pretrain:dirR251-20

        #run = '720'  # distance-matrix, l2-corr, dm-labels-weighted,

        run = 'abx'

        obj = 'rating'  # 'distance-matrix' 'rating' 'rating-size'

        rating_scale = 'none'
        reg_loss = None  # {'SampleCorrelation': 0.0}  # 'Dispersion', 'Std', 'FeatureCorrelation', 'SampleCorrelation'
        batch_size = 32

        epoch_pre = 20
        preload_weight = None  # FileManager.Weights('dirR', output_dir=input_dir).name(run='251c{}'.format(config), epoch=epoch_pre)

        model = DirectArch(miniXception_loader, input_shape, output_size=out_size, objective=obj, separated_prediction=False,
                           normalize=normalize, pooling='msrmac', l1_regularization=None, regularization_loss=reg_loss, batch_size=batch_size)
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
                    use_class_weight=False, use_confidence=True)
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
        #run = 'siam100'  # base model, res=0.5I, l2
        #run = 'siam101'  # base model, res=0.5I, l1 | b: lr->e-2
        #run = 'siam102'  # base model, res=0.5I, l2, msrmac
        #run = 'siam103'  # base model, res=0.5I, cosine
        #run = 'siam104'  # base model, res=0.5I, l1, norm-l1
        #run = 'siam200b'   # l2, max-pool, w256
        #run = 'siam202c'  # l2, rmac-pool, w512
        #run = 'siam203'  # l2, rmac-pool, w512, old data style
        #run = 'siam204'  # l2, msrmac-pool, w256
        #run = 'siam205'  # l1, max-pool, w256
        #run = 'siam206'  # l1, msrmac-pool, w256
        #run = 'siam207'  # l2, avg-pool, w256
        #run = '208c'  # l1, avg-pool, w256
        #run = '210b'  # l2, max-pool, alt loss 210:fix-attempt-at-loss, 210b:corrected-alt-loss
        #run = '214'  # l2, msrmac, alt loss (different)
        #run = '220' # l2, max, data-aug
        #run = '224'  # l2, msrmac, data-aug
        #run = '230' # l2 max, alt-loss, data-aug
        #run = '234'  # l2 msrmac, alt-loss, data-aug

        run = 'zzz'

        gen = True
        batch_size = 64 if local else 128

        # model
        generator = DataGeneratorSiam(data_loader,
                                      data_size=data_size, model_size=model_size, batch_size=batch_size,
                                      val_factor=0 if skip_validation else 3, balanced=True, objective="malignancy",
                                      do_augment=do_augment, augment=data_augment_params, weighted_rating=True,
                                      use_class_weight=False)

        model = SiamArch(miniXception_loader, input_shape, output_size=out_size, batch_size=batch_size,
                         distance='l2', normalize=normalize, pooling='msrmac')
        model.model.summary()
        model.compile(learning_rate=1e-3, decay=0)
        if gen:
            model.load_generator(generator)
        else:
            imgs_trn, lbl_trn = generator.next_train().__next__()
            imgs_val, lbl_val = generator.next_val().__next__()
            model.load_data(imgs_trn, lbl_trn, imgs_val, lbl_val)

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

        #run = '200'  # pretrain with dirR251-70

        #run = '300'   # obj: size
        #run = '311'  # obj: rating-size

        run = 'zzz'  #


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
