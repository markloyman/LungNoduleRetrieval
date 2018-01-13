from init import *
from Analysis import PredictRatings, RatingCorrelator


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
print("{} Set Analysis".format(post))
print('=' * 15)

start = timer()
try:

    pear_corr = []
    kend_corr = []
    for e in epochs:
        #pred, labels_test, meta = pickle.load(open(predict.pred_filename(run, epoch=e, post=post), 'br'))
        Reg = RatingCorrelator(PredictRatings.pred_filename(run, epoch=e, post=post))

        Reg.evaluate_embed_distance_matrix(method='euclidean')

        Reg.evaluate_rating_space()
        Reg.evaluate_rating_distance_matrix(method='euclidean')

        Reg.linear_regression()
        #Reg.scatter('embed', 'rating', xMethod="euclidean", yMethod='euclidean', sub=False)
        p, s, k = Reg.correlate('embed', 'rating')
        pear_corr.append(p)
        kend_corr.append(k)

    epochs = np.array(epochs)
    pear_corr = np.array(pear_corr)
    kend_corr = np.array(kend_corr)

    plt.figure()
    plt.plot(epochs, pear_corr)
    plt.plot(epochs, kend_corr)
    plt.title(run+'_'+ post)
    plt.xlabel('epochs')
    plt.ylabel('correlation')
    plt.legend(['pearson', 'kendall'])

    print('Plots Ready...')
    plt.show()

finally:
    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))