from init import *
from Analysis import RatingCorrelator
#from Network import PredictRating
#pred_loader = PredictRating()


def embed_correlate(network_type, run, post, epochs, rating_norm='none'):
    pear_corr = []
    kend_corr = []
    for e in epochs:
        # pred, labels_test, meta = pickle.load(open(loader.pred_filename(run, epoch=e, post=post), 'br'))
        file = FileManager.Embed(network_type)
        Reg = RatingCorrelator(file.name(run=run, epoch=e, dset=post))

        Reg.evaluate_embed_distance_matrix(method='euclidean')

        Reg.evaluate_rating_space(norm=rating_norm)
        Reg.evaluate_rating_distance_matrix(method='euclidean')

        Reg.linear_regression()
        # Reg.scatter('embed', 'rating', xMethod="euclidean", yMethod='euclidean', sub=False)
        p, s, k = Reg.correlate('embed', 'rating')
        pear_corr.append(p)
        kend_corr.append(k)

    epochs = np.array(epochs)
    pear_corr = np.array(pear_corr)
    kend_corr = np.array(kend_corr)

    plt.figure()
    plt.plot(epochs, pear_corr)
    plt.plot(epochs, kend_corr)
    plt.grid(which='major', axis='y')
    plt.title('embed_'+run + '_' + post)
    plt.xlabel('epochs')
    plt.ylabel('correlation')
    plt.legend(['pearson', 'kendall'])


def dir_rating_correlate(run, post, epochs, rating_norm='none'):
    pear_corr = []
    kend_corr = []
    for e in epochs:
        PredFile = FileManager.Pred(type='rating', pre='dirR')
        Reg = RatingCorrelator(PredFile(run=run, epoch=e, dset=post))

        Reg.evaluate_embed_distance_matrix(method='euclidean')

        Reg.evaluate_rating_space(norm=rating_norm)
        Reg.evaluate_rating_distance_matrix(method='euclidean')

        Reg.linear_regression()
        # Reg.scatter('embed', 'rating', xMethod="euclidean", yMethod='euclidean', sub=False)
        p, s, k = Reg.correlate('embed', 'rating')
        pear_corr.append(p)
        kend_corr.append(k)

    epochs = np.array(epochs)
    pear_corr = np.array(pear_corr)
    kend_corr = np.array(kend_corr)

    plt.figure()
    plt.plot(epochs, pear_corr)
    plt.plot(epochs, kend_corr)
    plt.grid(which='major', axis='y')
    plt.title('rating_'+run + '_' + post)
    plt.xlabel('epochs')
    plt.ylabel('correlation')
    plt.legend(['pearson', 'kendall'])


def dir_rating_rmse(run, post, epochs):
    PredFile = FileManager.Pred(type='rating', pre='dirR')
    images, predict, meta_data, labels, masks = PredFile.load(run=run, epoch=epochs[-1], dset=post)
    #images, predict, meta_data, labels, masks = pred_loader.load(run, epochs[-1], post)
    rating_property = ['Subtlety', 'Internalstructure', 'Calcification', 'Sphericity', 'Margin',
                       'Lobulation', 'Spiculation', 'Texture', 'Malignancy']
    for r in range(9):
        rmse = np.sqrt(np.mean((predict[:, r] - labels[:, r]) ** 2))
        print("{}: {:.2f}".format(rating_property[r], rmse))
    return None


def l2(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def dir_rating_view(run, post, epochs, factor=1.0):
    # load
    #images, predict, meta_data, labels, masks = pred_loader.load(run, epochs[-1], post)
    PredFile = FileManager.Pred(type='rating', pre='dirR')
    images, predict, meta_data, labels, masks = PredFile.load(run=run, epoch=epochs[-1], dset=post)
    # prepare
    images  = np.squeeze(images)
    labels  = (10*labels).astype('int')
    predict = (10*predict).astype('int')
    #plot
    select = [5, 23, 27, 51]
    plt.figure('view_'+run+'_'+post)
    for pid, i in enumerate(select):
        plt.subplot(2,2,pid+1)
        plt.imshow(images[i])
        plt.title(np.array2string(labels[i], prefix='L')+'\n'+np.array2string(predict[i], prefix='P'))
        plt.xticks([])
        plt.yticks([])
        if pid >= 0:
            dl = l2(labels[i], labels[select[0]])
            dp = l2(predict[i], predict[select[0]])
            plt.ylabel("{:.1f}\n{:.1f}".format(dl, dp))


run = '011'
#epochs = [10, 20, 30, 35, 40, 45, 50, 55] # [15, 20, 25, 30, 35, 39]
#run = '004'
#epochs = [15, 20, 25, 30, 35, 39]

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

    #dir_rating_correlate(run, post, epochs, rating_norm='none')
    #embed_correlate('dirR', run, post, epochs, rating_norm='none')
    #dir_rating_rmse(run, "Train", [25])
    dir_rating_rmse(run, "Valid", [55])

    #dir_rating_view(run, post, epochs, factor=10)

    print('=' * 15)
    print('Plots Ready...')
    plt.show()

finally:

    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))
