from Network.predict import Malignancy as PredictMal
from Network.predict import Rating as PredictRating

run = '201'
conf = 4
epochs = [40]

pooling = 'max'
rating_scale = 'none'

# 0     Test
# 1     Validation
# 2     Training
DataSubSet = 0

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

    if False:
        PredRating = PredictRating(pooling=pooling)
        for e in epochs:
            WeightsFile = FileManager.Weights('dirR').name(run, epoch=e)
            PredFile = FileManager.Pred(type='rating', pre='dirR')
            out_file = PredFile(run=run, epoch=e, dset=post)

            PredRating.predict_rating(WeightsFile, out_file, DataSubSet, rating_scale=rating_scale)
    else:
        PredMal = PredictMal(pooling=pooling)
        for e in epochs:
            WeightsFile = FileManager.Weights('dir').name(run, epoch=e)
            print("weight: {}".format(WeightsFile))
            PredFile = FileManager.Pred(type='malig', pre='dir')
            out_file = PredFile(run=run[:-1] + str(conf), epoch=e, dset=post)

            PredMal.data_size = 128
            PredMal.res = 0.5
            PredMal.predict_malignancy(WeightsFile, out_file, DataSubSet, configuration=conf)

    print('Ready...')

finally:
    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))