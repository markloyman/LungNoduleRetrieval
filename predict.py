from init import *
from Analysis import PredictRatings


run = '002'
epochs = [15, 20, 25, 30, 35, 40, 45, 50]

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

    for e in epochs:
        WeightsFile = FileManager.Weights('dirR').name(run, epoch=e)
        out_file = PredictRatings.pred_filename(run, epoch=e, post=post)
        PredictRatings.predict_rating(WeightsFile, out_file, DataSubSet)

    print('Ready...')

finally:
    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))