from init import *
from Network import PredictRating

run = '010'
epochs = [10, 20, 30, 35, 40, 45, 50, 55]

pooling = 'rmac'
rating_scale = 'Scale'

# 0     Test
# 1     Validation
# 2     Training
DataSubSet = 2

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

    PredRating = PredictRating(pooling=pooling)
    for e in epochs:
        WeightsFile = FileManager.Weights('dirR').name(run, epoch=e)
        PredFile = FileManager.Pred(type='rating', pre='dirR')
        out_file = PredFile(run=run, epoch=e, post=post)

        PredRating.predict_rating(WeightsFile, out_file, DataSubSet, rating_scale=rating_scale)

    print('Ready...')

finally:
    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))