import gc
from keras import backend as K

from Network.DataGenSiam import DataGenerator
from Network.model import miniXception_loader
from Network.siameseArch import siamArch

## --------------------------------------- ##
## ------- General Setup ----------------- ##
## --------------------------------------- ##

data_size  = 144
model_size = 128
res    = 'Legacy' #0.7
sample = 'Normal' #'Uniform'
input_shape = (model_size, model_size, 1)

out_size = 128

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

    #run = 'siam000'   # chained, margin = 5, batch = 64,
    #run = 'siam001'    # base, margin = 5, batch = 64
    #run = 'siam002' # overlapped
    #run = 'siam000b'  # CHAINED, x2 sets
    #run = 'siam003'  # CHAINED, decrease learning rate to 1e-4
    #run = 'siam005'  # CHAINED, Sample-Weight
    #run = 'siam00YY'  # with class w 124
    #run = 'siam00ZZ'   # no class w
    #run = 'siam00YYY'  # with class w 126
    #run = 'siam006'  # with class w 126, uniform data norm (+dicom rescale)
    #run = 'siam007XXX'  # siam with l1 distance
    # new dataset 144,0.5
    #run = 'siam008'  # data-aug, no class weight
    #run = 'siam009'  # no data-aug, no class weight
    #run = 'siam010X'  # legacy data, no data-aug, no class weight

    # fully data regen

    #run = 'siam011X' # margin=1, 128-Legacy-Normal, l2, lr=1e-3 (decay=0.1), batch=64, no data-aug, no sample-w
    #run = 'siam011XX'  # margin=5, 128-Legacy-Normal, l2, lr=1e-3 (decay=0.1), batch=64, no data-aug, no sample-w
    #run = 'siam011XXX'  # margin=5, 128-Legacy-Normal, l2, lr=1e-3 (decay=0), batch=64, no data-aug, no sample-w [BINGO]
    #run = 'siam012'  # margin=5, 128-Legacy-Normal, l2, lr=1e-3 (decay=0), batch=64, no data-aug, batch-wise sample-w
    #run = 'siam013'  # margin=5, 128-Legacy-Uniform, l2, lr=1e-3 (decay=0), batch=64, no data-aug, no sample-w
    #run = 'siam014'  # margin=5, 128-0.5-Normal, l2, lr=1e-3 (decay=0), batch=64, no data-aug, no sample-w
    #run = 'siam015'  # margin=5, 128-0.7-Normal, l2, lr=1e-3 (decay=0), batch=64, no data-aug, no sample-w
    #run = 'siam016'  # margin=5, 144-Legacy-Normal, + data-aug, no sample-w

    # fixed num of batches in DataGen

    #run = 'siam017'  # margin=5, 144-Legacy-Normal, + data-aug (training ONLY), no sample-w
    #run = 'siam018'  # margin=5, 128-Legacy-Normal, + no data-aug, sample-w (global w, method='count')
    #run = 'siam019X'  # margin=5, 128-Legacy-Normal, + no data-aug, sample-w (global w, method='dummy')

    # removed unneeded resize that fucked up the image values

    #run = 'siam020X'  # margin=5, 128-Legacy-Normal, l2, lr=1e-3 (decay=0), batch=64, no data-aug, no sample-w
    #run = 'siam021'  # margin=1, 128-Legacy-Normal, l2, lr=1e-3 (decay=0), batch=64, no data-aug, no sample-w

    #run = 'siam022'  # output_size=512, margin=5, 128-Legacy-Normal, l2, lr=1e-3 (decay=0), batch=64, no data-aug, no sample-w
    #run = 'siam023'  # pooling=max, output_size=512, margin=5, 128-Legacy-Normal, l2, lr=1e-3 (decay=0), batch=64, no data-aug, no sample-w
    #run = 'siam024'  # pooling=max, output_size=512, margin=10, 128-Legacy-Normal, l2, lr=1e-3 (decay=0), batch=64, no data-aug, no sample-w

    #run = 'siam025'  # l2 normalize, pooling=max, output_size=512, margin=1, 128-Legacy-Normal, l2, lr=1e-3 (decay=0), batch=64, no data-aug, no sample-w
    #run = 'siam026'  # l2 normalize, pooling=rmac, output_size=128, margin=1, 128-Legacy-Normal, l2, lr=1e-3 (decay=0), batch=64, no data-aug, no sample-w
    #run = 'siam026' # norm off, margin=5

    #run = 'siam027' # l2 normalize (axis=-1), pooling=rmac, out=128, margin=1
    #run = 'siam028'  # l2 normalize (axis=-1), pooling=rmac, out=128, margin=1, sample-w dummy
    #run = 'siam029'   # l2 normalize (axis=-1), pooling=rmac, out=128, margin=1, sample-w count

    #run = 'siam030'  # l2 normalize (axis=-1), pooling=rmac, out=128, margin=1, data-agum ('crop_stdev': 0.1), sample-w count
    #run = 'siam031'  # l2 normalize (axis=-1), pooling=rmac, out=128, margin=1, data-agum ('crop_stdev': 0.05), sample-w count
    #run = 'siam032'  ????? # l2 normalize (axis=-1), pooling=rmac, out=128, margin=1, data-agum ('crop_stdev': 0.05), sample-w count

    run = 'siam033X'  # l2 normalize (axis=-1), pooling=rmac, out=128, margin=1, data-agum ('crop_stdev': 0.0), sample-w count

    #run = 'siam034'  # l2 normalize (axis=-1), pooling=rmac, out=128, margin=1, data-agum {'max_angle': 15, 'flip_ratio': 0.2, 'crop_stdev': 0.15}, sample-w count

    # model
    data_augment_params = {'max_angle': 0, 'flip_ratio': 0.0, 'crop_stdev': 0.0}

    generator = DataGenerator(data_size=data_size, model_size=model_size, res=res, sample=sample, batch_sz=64,
                              do_augment=True, augment=data_augment_params,
                              use_class_weight=True, class_weight='count')

    model = siamArch(miniXception_loader, input_shape, distance='l2', output_size=out_size, normalize=True)
    model.model.summary()
    model.compile(learning_rate=1e-3, decay=0)

    model.load_generator(generator)

    model.train(label=run, n_epoch=50, gen=True)

    K.clear_session()
    gc.collect()