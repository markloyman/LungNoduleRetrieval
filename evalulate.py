from init import *
from Analysis import RatingCorrelator
#from Network import PredictRating
#pred_loader = PredictRating()


def accuracy(true, pred):
    #pred = np.clip(pred, 0, 1)
    pred = np.squeeze(np.round(pred).astype('uint'))
    mask = (true==pred).astype('uint')
    acc = np.mean(mask)
    return acc


def embed_correlate(network_type, run, post, epochs, rating_norm='none'):
    pear_corr = []
    kend_corr = []
    for e in epochs:
        # pred, labels_test, meta = pickle.load(open(loader.pred_filename(run, epoch=e, post=post), 'br'))
        file = FileManager.Embed(network_type)
        Reg = RatingCorrelator(file.name(run=run, epoch=e, dset=post))

        Reg.evaluate_embed_distance_matrix(method='euclidean', round=(rating_norm=='Round'))

        Reg.evaluate_rating_space(norm=rating_norm)
        Reg.evaluate_rating_distance_matrix(method='euclidean')

        Reg.linear_regression()
        # Reg.scatter('embed', 'rating', xMethod="euclidean", yMethod='euclidean', sub=False)
        p, s, k = Reg.correlate_retrieval('embed', 'rating')
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

        Reg.evaluate_embed_distance_matrix(method='euclidean', round=(rating_norm=='Round'))

        Reg.evaluate_rating_space(norm=rating_norm)
        Reg.evaluate_rating_distance_matrix(method='euclidean')

        Reg.linear_regression()
        # Reg.scatter('embed', 'rating', xMethod="euclidean", yMethod='euclidean', sub=False)
        p, s, k = Reg.correlate_retrieval('embed', 'rating', round=(rating_norm=='Round'))
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
    #images, predict, meta_data, labels, masks = pred_loader.load(run, epochs[-1], post)
    rating_property = ['Subtlety', 'Internalstructure', 'Calcification', 'Sphericity', 'Margin',
                       'Lobulation', 'Spiculation', 'Texture', 'Malignancy']
    PredFile = FileManager.Pred(type='rating', pre='dirR')
    R = np.zeros([len(epochs), 10])
    for i, e in enumerate(epochs):
        print(" Epoch {}:".format(e))
        images, predict, meta_data, labels, masks = PredFile.load(run=run, epoch=e, dset=post)
        for r in range(9):
            rmse = np.sqrt(np.mean((predict[:, r] - labels[:, r]) ** 2))
            print("\t{}: \t{:.2f}".format(rating_property[r], rmse))
            R[i, r] = rmse
        rmse = np.sqrt(np.mean(np.sum((predict - labels) ** 2, axis=1)))
        print("\t{}: \t{:.2f}".format(rating_property[r], rmse))
        R[i, 9] = rmse
    plt.figure()
    plt.title('Rating RMSE')
    plt.plot(epochs, R)
    plt.legend(rating_property+['Overall'])

    return R


def dir_rating_delta_rmse(run, post, e0, e1):
    PredFile = FileManager.Pred(type='rating', pre='dirR')
    images0, predict0, meta_data0, labels0, masks0 = PredFile.load(run=run, epoch=e0, dset=post)
    images1, predict1, meta_data1, labels1, masks1 = PredFile.load(run=run, epoch=e1, dset=post)
    rmse0 = np.sqrt(np.sum((predict0 - labels0) ** 2, axis=1))
    rmse1 = np.sqrt(np.sum((predict1 - labels1) ** 2, axis=1))

    delta = rmse1 - rmse0

    return delta


def dir_rating_accuracy(run, post, epochs):
    #images, predict, meta_data, labels, masks = pred_loader.load(run, epochs[-1], post)
    rating_property = ['Subtlety', 'Internalstructure', 'Calcification', 'Sphericity', 'Margin',
                       'Lobulation', 'Spiculation', 'Texture', 'Malignancy']
    PredFile = FileManager.Pred(type='rating', pre='dirR')
    acc = np.zeros([len(epochs), 1])
    for i, e in enumerate(epochs):
        images, predict, meta_data, labels, masks = PredFile.load(run=run, epoch=e, dset=post)
        acc[i] = accuracy(labels, predict)
    plt.figure()
    plt.title('Rating Acc')
    plt.plot(epochs, acc)

    return acc


def l2(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def dir_rating_view(run, post, epochs, factor=1.0):
    # load
    #images, predict, meta_data, labels, masks = pred_loader.load(run, epochs[-1], post)
    PredFile = FileManager.Pred(type='rating', pre='dirR')
    images, predict, meta_data, labels, masks = PredFile.load(run=run, epoch=epochs[-1], dset=post)
    # prepare
    images  = np.squeeze(images)
    labels  = np.round(factor*labels).astype('int')
    predict = np.round(factor*predict).astype('int')
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


if __name__ == "__main__":
    run = '011X'
    epochs = [15, 25, 35, 45, 55, 65, 75, 85, 95]

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
    print("{} Set Analysis".format(post))
    print('=' * 15)

    start = timer()
    try:

        #dir_rating_correlate(run, post, epochs, rating_norm='Round')
        #embed_correlate('dirR', run, post, epochs, rating_norm='Round')
        #dir_rating_rmse(run, post, epochs)
        dir_rating_accuracy(run, post, epochs)

        #dir_rating_view(run, post, epochs, factor=1)

        print('=' * 15)
        print('Plots Ready...')
        plt.show()

    finally:

        total_time = (timer() - start) / 60 / 60
        print("Total runtime is {:.1f} hours".format(total_time))
