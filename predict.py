from Network.predict import Malignancy as PredictMal
from Network.predict import Rating as PredictRating
from init import *
from Network import FileManager


run = '200'
conf = None
epochs = np.arange(1, 100)

pooling = 'max'
rating_scale = 'none'

# 0     Test
# 1     Validation
# 2     Training
DataSubSet = 1

if DataSubSet == 0:
    post = "Test"
elif DataSubSet == 1:
    post = "Valid"
elif DataSubSet == 2:
    post = "Train"
else:
    assert False
print("{} Prediction".format(post))
print('=' * 15)

start = timer()
try:

    if conf is not None:
        run += 'c{}'.format(conf)

    if True:
        PredRating = PredictRating(pooling=pooling)
        PredRating.load_dataset(data_subset_id=DataSubSet, size=128, rating_scale=rating_scale, configuration=conf)
        preds = []
        valid_epochs = []
        for e in epochs:
            WeightsFile = FileManager.Weights('dirR').name(run, epoch=e)
            PredFile = FileManager.Pred(type='rating', pre='dirR')
            out_file = PredFile(run=run, dset=post)

            data, out_filename = PredRating.predict_rating(WeightsFile, out_file)
            images_test, pred, meta_test, classes_test, labels_test, masks_test = data
            preds.append(np.expand_dims(pred, axis=0))
        preds = np.concatenate(preds, axis=0)
        pickle.dump((preds, np.array(epochs), images_test, meta_test, classes_test, labels_test, masks_test), open(out_filename, 'bw'))
    else:
        PredMal = PredictMal(pooling=pooling)
        for e in epochs:
            WeightsFile = FileManager.Weights('dir').name(run, epoch=e)
            print("weight: {}".format(WeightsFile))
            PredFile = FileManager.Pred(type='malig', pre='dir')
            out_file = PredFile(run=run[:-1] + str(conf), dset=post)

            PredMal.data_size = 128
            PredMal.res = 0.5
            PredMal.predict_malignancy(WeightsFile, out_file, DataSubSet, configuration=conf)

    print('Ready...')

finally:
    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))