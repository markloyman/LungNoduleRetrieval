from Analysis import RatingCorrelator
from Analysis.analysis import smooth
from init import *
from Network import FileManager

#from Network import PredictRating
#pred_loader = PredictRating()

alpha = 0.4

def accuracy(true, pred):
    #pred = np.clip(pred, 0, 1)
    pred = np.squeeze(np.round(pred).astype('uint'))
    true = np.squeeze(np.round(true).astype('uint'))
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


def dir_rating_correlate(run, post, epochs, rating_norm='none', n_groups=5):
    pear_corr = [[] for i in range(n_groups)]
    kend_corr = [[] for i in range(n_groups)]
    plot_data_filename = './Plots/Data/rating_correlation_{}{}.p'.format('dirR', run)
    try:
        pear_corr, kend_corr = pickle.load(open(plot_data_filename, 'br'))
        print("Loaded results for {}".format(run))
    except:
        print("Evaluating Rating Correlation for {}".format(run))
        for c, run_config in enumerate([run + 'c{}'.format(config) for config in range(n_groups)]):
            PredFile = FileManager.Pred(type='rating', pre='dirR')
            Reg = RatingCorrelator(PredFile(run=run_config, dset=post), multi_epoch=True)
            for e in epochs:
                Reg.evaluate_embed_distance_matrix(method='euclidean', epoch=e, round=(rating_norm == 'Round'))
                Reg.evaluate_rating_space(norm=rating_norm)
                Reg.evaluate_rating_distance_matrix(method='euclidean')

                Reg.linear_regression()
                # Reg.scatter('embed', 'rating', xMethod="euclidean", yMethod='euclidean', sub=False)
                p, s, k = Reg.correlate_retrieval('embed', 'rating', round=(rating_norm == 'Round'), verbose=False)
                pear_corr[c].append(p)
                kend_corr[c].append(k)

            pear_corr[c] = np.array(pear_corr[c])
            kend_corr[c] = np.array(kend_corr[c])

        pear_corr = np.mean(pear_corr, axis=0)
        kend_corr = np.mean(kend_corr, axis=0)
        pickle.dump((pear_corr, kend_corr), open(plot_data_filename, 'bw'))

    pear_corr = smooth(pear_corr[:, 0]), smooth(pear_corr[:, 1])
    kend_corr = smooth(kend_corr[:, 0]), smooth(kend_corr[:, 1])
    epochs = np.array(epochs)

    plt.figure('Rating2Rating:' + run + '-' + post)
    q = plt.plot(epochs, pear_corr[0])
    plt.plot(epochs, pear_corr[0] + pear_corr[1], color=q[0].get_color(), ls='--', alpha=alpha)
    plt.plot(epochs, pear_corr[0] - pear_corr[1], color=q[0].get_color(), ls='--', alpha=alpha)

    q = plt.plot(epochs, kend_corr[0])
    plt.plot(epochs, kend_corr[0] + kend_corr[1], color=q[0].get_color(), ls='--', alpha=alpha)
    plt.plot(epochs, kend_corr[0] - kend_corr[1], color=q[0].get_color(), ls='--', alpha=alpha)

    plt.grid(which='major', axis='y')
    plt.title('rating_'+run + '_' + post)
    plt.xlabel('epochs')
    plt.ylabel('correlation')
    plt.legend(['pearson', '', '', 'kendall', '', ''])


def dir_rating_rmse(run, post, epochs, n_groups=5):
    #images, predict, meta_data, labels, masks = pred_loader.load(run, epochs[-1], post)
    rating_property = ['Subtlety', 'Internalstructure', 'Calcification', 'Sphericity', 'Margin',
                       'Lobulation', 'Spiculation', 'Texture', 'Malignancy']

    plot_data_filename = './Plots/Data/rmse_{}{}.p'.format('dirR', run)
    try:
        R = pickle.load(open(plot_data_filename, 'br'))
        print("Loaded results for {}".format(run))
    except:
        print("Evaluating RMSE for {}".format(run))
        PredFile = FileManager.Pred(type='rating', pre='dirR')
        R = np.zeros([len(epochs), 10, n_groups])

        for c, run_config in enumerate([run + 'c{}'.format(config) for config in range(n_groups)]):
            predict, valid_epochs, images, meta_data, classes, labels, masks = PredFile.load(run=run_config, dset=post)
            for i, e in enumerate(epochs):
                print(" Epoch {}:".format(e))
                try:
                    idx = int(np.argwhere(valid_epochs == e))
                except:
                    print('skip epoch {}'.format(e))
                    continue
                pred = predict[idx]
                for r in range(9):
                    rmse = np.sqrt(np.mean((pred[:, r] - labels[:, r]) ** 2))
                    print("\t{}: \t{:.2f}".format(rating_property[r], rmse))
                    R[i, r] = rmse
                rmse = np.sqrt(np.mean(np.sum((pred - labels) ** 2, axis=1)))
                print("\t{}: \t{:.2f}".format(rating_property[r], rmse))
                R[i, 9, c] = rmse
        R = np.mean(R, axis=2)
        pickle.dump(R, open(plot_data_filename, 'bw'))

    # smooth
    for r in range(9):
        R[:, r] = smooth(R[:, r])
    plt.figure('RMSE ' + run + '-' + post)
    plt.title('Rating RMSE')
    plt.plot(epochs, R)
    plt.legend(rating_property+['Overall'])
    plt.grid(True, axis='y')

    return R


def dir_rating_params_correlate(run, post, epochs, rating_norm='none', n_groups=5):

    reference = [0.7567, 0, 0, 0.5945, 0.7394, 0.5777, 0.6155, 0.7445, 0.6481]
    rating_property = ['Subtlety', 'Internalstructure', 'Calcification', 'Sphericity', 'Margin',
                       'Lobulation', 'Spiculation', 'Texture', 'Malignancy']

    pear_corr = [[] for i in range(n_groups)]
    plot_data_filename = './Plots/Data/rating_params_correlation_{}{}.p'.format('dirR', run)
    try:
        pear_corr = pickle.load(open(plot_data_filename, 'br'))
        print("Loaded results for {}".format(run))
    except:
        print("Evaluating Rating Correlation for {}".format(run))
        for c, run_config in enumerate([run + 'c{}'.format(config) for config in range(n_groups)]):
            PredFile = FileManager.Pred(type='rating', pre='dirR')
            Reg = RatingCorrelator(PredFile(run=run_config, dset=post), multi_epoch=True)
            Reg.evaluate_rating_space(norm=rating_norm)
            #valid_epochs = []
            for e in epochs:
                p = Reg.correlate_to_ratings(epoch=e, round=(rating_norm == 'Round'))
                if not np.all(np.isfinite(p)):
                    print('nan at: conf={}, epoch={}'.format(c, e))
                pear_corr[c].append(p)
                #valid_epochs.append(e)

            pear_corr[c] = np.array(pear_corr[c])

        pear_corr = np.mean(pear_corr, axis=0)
        #pickle.dump(pear_corr, open(plot_data_filename, 'bw'))

    for p in range(pear_corr.shape[1]):
        pear_corr[:, p] = smooth(pear_corr[:, p])
    epochs = np.array(epochs)

    plt.figure('Rating2Rating:' + run + '-' + post)
    q = plt.plot(epochs, pear_corr)
    for line, ref in zip(q, reference):
        plt.plot(epochs, ref * np.ones_like(epochs), color=line.get_color(), ls='--', alpha=0.6)

    plt.grid(which='major', axis='y')
    plt.title('rating_'+run + '_' + post)
    plt.xlabel('epochs')
    plt.ylabel('correlation')
    plt.legend(rating_property)


def dir_rating_delta_rmse(run, post, e0, e1):
    PredFile = FileManager.Pred(type='rating', pre='dirR')
    images0, predict0, meta_data0, labels0, masks0 = PredFile.load(run=run, epoch=e0, dset=post)
    images1, predict1, meta_data1, labels1, masks1 = PredFile.load(run=run, epoch=e1, dset=post)
    rmse0 = np.sqrt(np.sum((predict0 - labels0) ** 2, axis=1))
    rmse1 = np.sqrt(np.sum((predict1 - labels1) ** 2, axis=1))

    delta = rmse1 - rmse0

    return delta


def dir_rating_accuracy(run, post, epochs, n_groups=5):
    #images, predict, meta_data, labels, masks = pred_loader.load(run, epochs[-1], post)
    rating_property = ['Subtlety', 'Internalstructure', 'Calcification', 'Sphericity', 'Margin',
                       'Lobulation', 'Spiculation', 'Texture', 'Malignancy']
    PredFile = FileManager.Pred(type='rating', pre='dirR')
    acc = np.zeros([len(epochs), n_groups])
    for c, run_config in enumerate([run + 'c{}'.format(config) for config in range(n_groups)]):
        predict, valid_epochs, images, meta_data, classes, labels, masks = PredFile.load(run=run_config, dset=post)
        for i, e in enumerate(epochs):
            try:
                idx = int(np.argwhere(valid_epochs == e))
            except:
                print('skip epoch {}'.format(e))
                continue
            acc[i, c] = accuracy(labels, predict[idx])
    acc = np.mean(acc, axis=2)
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
    run = '223'
    epochs = np.arange(1, 81)  # [1, 10, 20, 30]

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

        #dir_rating_correlate(run, post, epochs, rating_norm='Round')
        #embed_correlate('dirR', run, post, epochs, rating_norm='Round')
        #dir_rating_accuracy(run+'c{}'.format(config), post, epochs)
        dir_rating_params_correlate(run, post, epochs, rating_norm='none')  # rating_norm='Round'
        dir_rating_rmse(run, post, epochs)

        #dir_rating_view(run, post, epochs, factor=1)

        print('=' * 15)
        print('Plots Ready...')
        plt.show()

    finally:

        total_time = (timer() - start) / 60 / 60
        print("Total runtime is {:.1f} hours".format(total_time))
