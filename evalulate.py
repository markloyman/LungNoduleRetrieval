from Analysis import RatingCorrelator
from Analysis.analysis import smooth
from init import *
from Network import FileManager
from experiments import CrossValidationManager

alpha = 0.4


def accuracy(true, pred):
    mask = (np.abs(true-pred) <=1).astype('uint')
    acc = np.mean(mask)
    return acc

'''
def accuracy(true, pred):
    #pred = np.clip(pred, 0, 1)
    pred = np.squeeze(np.round(pred).astype('uint'))
    true = np.squeeze(np.round(true).astype('uint'))
    mask = (true==pred).astype('uint')
    acc = np.mean(mask)
    return acc
'''

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


def dir_rating_correlate(run, post, epochs, rating_norm='none',  clustered_rating_distance=True, n_groups=5):
    pear_corr = [[] for i in range(n_groups)]
    kend_corr = [[] for i in range(n_groups)]
    plot_data_filename = './Plots/Data/rating_correlation_{}{}.p'.format('dirR', run)
    try:
        print('SKIPING')
        assert False
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
                Reg.evaluate_rating_distance_matrix(method='euclidean', clustered_rating_distance=clustered_rating_distance)

                Reg.linear_regression()
                # Reg.scatter('embed', 'rating', xMethod="euclidean", yMethod='euclidean', sub=False)
                p, s, k = Reg.correlate_retrieval('embed', 'rating', round=(rating_norm == 'Round'), verbose=False)
                pear_corr[c].append(p)
                kend_corr[c].append(k)

            pear_corr[c] = np.array(pear_corr[c])
            kend_corr[c] = np.array(kend_corr[c])

        pear_corr = np.mean(pear_corr, axis=0)
        kend_corr = np.mean(kend_corr, axis=0)
        print('NO DUMP')
        #pickle.dump((pear_corr, kend_corr), open(plot_data_filename, 'bw'))

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


def dir_rating_rmse(run, post, epochs, net_type, dist='RMSE', weighted=False, configurations=list(range(5)), USE_CACHE=True, DUMP=True):
    #images, predict, meta_data, labels, masks = pred_loader.load(run, epochs[-1], post)
    rating_property = ['Subtlety', 'Internalstructure', 'Calcification', 'Sphericity', 'Margin',
                       'Lobulation', 'Spiculation', 'Texture', 'Malignancy']

    plot_data_filename = './Plots/Data/{}_{}{}.p'.format(dist, net_type, run)
    try:
        if USE_CACHE is False:
            print("skipping...")
            assert False
        R = pickle.load(open(plot_data_filename, 'br'))
        print("Loaded results for {}".format(run))
    except:
        print("Evaluating RMSE for {}".format(run))
        PredFile = FileManager.Pred(type='rating', pre=net_type)
        R = np.zeros([len(epochs), 10, len(configurations)])

        for c, run_config in enumerate([run + 'c{}'.format(config) for config in configurations]):
            predict, valid_epochs, images, meta_data, classes, labels, masks, conf, rating_weights, z = PredFile.load(run=run_config, dset=post)
            labels = np.array([np.mean(l, axis=0) for l in labels])
            for i, e in enumerate(epochs):
                #print("=" * 20)
                #print(" Epoch {}:".format(e))
                #print("-" * 20)
                try:
                    idx = int(np.argwhere(valid_epochs == e))
                except:
                    print('skip epoch {}'.format(e))
                    continue
                pred = predict[idx]

                for r, max_val in zip(range(9), [5, 5, 6, 5, 5, 5, 5, 5, 5]):
                    #print("{}:".format(rating_property[r]))
                    W = np.ones(labels.shape[0])
                    if weighted:
                        w = np.histogram(labels[:, r], bins=np.array(range(max_val+1))+0.5)[0]
                        #print("\tcounts - {}".format(w))
                        w = 1 - w / np.sum(w)
                        w /= (len(w) - 1)
                        assert np.abs(w.sum() - 1) < 1e-6
                        #print("\tweighted by {}".format(w))
                        #pred_w = np.minimum(np.maximum(pred[:, r], 1.0), max_val)
                        W = w[np.round(labels[:, r] - 1).astype('int')]
                    if dist=='RMSE':
                        err = W.dot((pred[:, r] - labels[:, r])**2)
                        err = np.sqrt(err/np.sum(W))
                    elif dist=='ABS':
                        err = W.dot(np.abs(pred[:, r] - labels[:, r])) / np.sum(W)
                    else:
                        print('{} unrecognized distance'.format(dist))
                        assert False
                    #print("rmse: \t{:.2f}".format(err))
                    R[i, r, c] = err
                rmse = np.sqrt(np.mean(np.sum((pred - labels) ** 2, axis=1)))
                #print("=" * 20)
                #print("overall: \t{:.2f}".format(rmse))
                R[i, 9, c] = rmse
        R = np.mean(R, axis=2)
        for i, e in enumerate(epochs):
            print("=" * 20)
            print(" Epoch {}:".format(e))
            print("-" * 20)
            for p, property in enumerate(rating_property):
                print("\t{}: \t{:.2f}".format(property, R[i, p]))
            print("\t" + ("-" * 10))
            print("\toverall: \t{:.2f}".format(R[i, 9]))

        if DUMP:
            pickle.dump(R, open(plot_data_filename, 'bw'))
        else:
            print("No Dump")

    # smooth
    for r in range(9):
        R[:, r] = smooth(R[:, r])
    plt.figure(dist + ' ' + run + '-' + post)
    plt.title('Rating ' + dist)
    plt.plot(epochs, R)
    plt.legend(rating_property+['Overall'])
    plt.grid(True, axis='y')

    return R


def dir_rating_params_correlate(run, post, epochs, net_type, rating_norm='none', configurations=list(range(5)), USE_CACHE=True, DUMP=True):

    reference = [0.7567, 0.5945, 0.7394, 0.5777, 0.6155, 0.7445, 0.6481]  # 0, 0,
    rating_property = ['Subtlety', 'Sphericity', 'Margin',
                       'Lobulation', 'Spiculation', 'Texture', 'Malignancy']  # 'Internalstructure', 'Calcification',
    mask = [True, False, False, True, True, True, True, True, True]

    pear_corr = [[] for i in configurations]
    plot_data_filename = './Plots/Data/rating_params_correlation_{}{}.p'.format(net_type, run)
    try:
        if USE_CACHE is False:
            print('SKIPPING')
            assert False
        pear_corr = pickle.load(open(plot_data_filename, 'br'))
        print("Loaded results for {}".format(run))
    except:
        print("Evaluating Rating Correlation for {}".format(run))
        for c, run_config in enumerate([run + 'c{}'.format(config) for config in configurations]):
            PredFile = FileManager.Pred(type='rating', pre=net_type)
            Reg = RatingCorrelator(PredFile(run=run_config, dset=post), multi_epoch=True, conf=c)
            Reg.evaluate_rating_space(norm=rating_norm)
            #valid_epochs = []
            for e in epochs:
                p = Reg.correlate_to_ratings(epoch=e, round=(rating_norm == 'Round'))
                if not np.all(np.isfinite(p[mask])):
                    print('nan at: conf={}, epoch={}'.format(c, e))
                pear_corr[c].append(p[mask])
                #valid_epochs.append(e)

            pear_corr[c] = np.array(pear_corr[c])

        pear_corr = np.mean(pear_corr, axis=0)
        if DUMP:
            pickle.dump(pear_corr, open(plot_data_filename, 'bw'))
        else:
            print('NO DUMP')

    for i, e in enumerate(epochs):
        print("=" * 20)
        print(" Epoch {}:".format(e))
        print("-" * 20)
        for p, property in enumerate(rating_property):
            print("\t{}: \t{:.2f}".format(property, pear_corr[i, p]))
        #print("\t" + ("-" * 10))
        #print("\toverall: \t{:.2f}".format(R[i, 9]))

    for p in range(pear_corr.shape[1]):
        pear_corr[:, p] = smooth(pear_corr[:, p], window_length=5, polyorder=2)
    epochs = np.array(epochs)

    plt.figure('RatingParams2Rating:' + run + '-' + post)
    q = plt.plot(epochs, pear_corr, linewidth=2.5)
    for line, ref in zip(q, reference):
        plt.plot(epochs, ref * np.ones_like(epochs), color=line.get_color(), ls='--', linewidth=4, alpha=0.6)

    plt.grid(which='major', axis='y')
    plt.title('rating_'+run + '_' + post)
    plt.xlabel('epochs')
    plt.ylabel('correlation')
    plt.legend(rating_property)


def dir_rating_accuracy(run, post, net_type, epochs, n_groups=5):
    #images, predict, meta_data, labels, masks = pred_loader.load(run, epochs[-1], post)
    rating_property = ['Subtlety', 'Internalstructure', 'Calcification', 'Sphericity', 'Margin',
                       'Lobulation', 'Spiculation', 'Texture', 'Malignancy']
    PredFile = FileManager.Pred(type='rating', pre=net_type)
    acc = np.zeros([len(epochs), n_groups, len(rating_property)])
    for c, run_config in enumerate([run + 'c{}'.format(config) for config in range(n_groups)]):
        predict, valid_epochs, images, meta_data, classes, labels, masks = PredFile.load(run=run_config, dset=post)
        labels = np.array([np.mean(t, axis=0) for t in labels])
        for i, e in enumerate(epochs):
            try:
                idx = int(np.argwhere(valid_epochs == e))
            except:
                print('skip epoch {}'.format(e))
                continue
            for ridx, r in enumerate(rating_property):
                acc[i, c, ridx] = accuracy(labels[:,ridx], predict[idx,:,ridx])
    acc = np.mean(acc, axis=1)
    plt.figure()
    plt.title('Rating Acc')
    plt.plot(epochs, acc)
    plt.legend(rating_property)

    return acc


def dir_size_rmse(run, post, epochs, net_type, dist='RMSE', weighted=False, n_groups=5):

    plot_data_filename = './Plots/Data/size{}_{}{}.p'.format(dist, net_type, run)
    try:
        assert False
        R = pickle.load(open(plot_data_filename, 'br'))
        print("Loaded results for {}".format(run))
    except:
        print("Evaluating Size RMSE for {}".format(run))
        PredFile = FileManager.Pred(type='size', pre=net_type)
        R = np.zeros([len(epochs), n_groups])

        for c, run_config in enumerate([run + 'c{}'.format(config) for config in range(n_groups)]):
            predict, valid_epochs, images, meta_data, classes, labels, masks = PredFile.load(run=run_config, dset=post)
            labels = np.array(labels)
            for i, e in enumerate(epochs):
                print(" Epoch {}:".format(e))
                try:
                    idx = int(np.argwhere(valid_epochs == e))
                except:
                    print('skip epoch {}'.format(e))
                    continue
                pred = predict[idx]
                '''
                W = np.ones(labels.shape[0])
                if weighted:
                    assert False
                    w = np.histogram(labels[:, r], bins=np.array(range(64))+0.5)[0]
                    w = 1 - w / np.sum(w)
                    pred_w = np.minimum(np.maximum(pred[:, r], 1.0), max_val)
                    W = w[np.round(pred_w - 1).astype('int')]
                if dist=='RMSE':
                    err = W.dot((pred - labels)**2)
                    err = np.sqrt(err/np.sum(W))
                elif dist=='ABS':
                    err = W.dot(np.abs(pred - labels)) / np.sum(W)
                else:
                    print('{} unrecognized distance'.format(dist))
                    assert False
                '''
                rmse = np.sqrt(np.mean(np.sum((pred - labels) ** 2, axis=1)))
                R[i, c] = rmse
        R = np.mean(R, axis=1)
        pickle.dump(R, open(plot_data_filename, 'bw'))

    # smooth
    R = smooth(R)

    plt.figure(dist + ' ' + net_type + run + '-' + post)
    plt.title('Size ' + dist)
    plt.plot(epochs, R)
    #plt.legend(rating_property+['Overall'])
    plt.grid(True, axis='y')

    return R


def l2(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def dir_rating_view(run, post, epochs, net_type='dirR', factor=1.0):
    # load
    #images, predict, meta_data, labels, masks = pred_loader.load(run, epochs[-1], post)
    PredFile = FileManager.Pred(type='rating', pre=net_type)

    predict, epochs, meta_data, images, classes, labels, masks, _, _, _ = PredFile.load(run=run+'c0', dset=post)
    # prepare
    images  = np.squeeze(images)
    labels = np.array([np.mean(l, axis=0) for l in labels])
    labels  = np.round(factor*labels).astype('int')
    predict = np.round(factor*predict[-1]).astype('int')
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
    #run = '251', '300'
    run = '813'  # '251'  # '251' '512c'  # '412'
    net_type = 'dirR'
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

    # confs = range(5)
    # confs = ['01']
    confs = [CrossValidationManager('PRED').get_run_id(i) for i in range(10)]

    start = timer()
    try:
        #dir_rating_correlate(run, post, epochs, rating_norm='none', clustered_rating_distance=True)
        #embed_correlate('dirR', run, post, epochs, rating_norm='Round')
        #dir_rating_accuracy(run, post, net_type, epochs)
        dir_rating_params_correlate(run, post, epochs, net_type=net_type, rating_norm='none', configurations=confs)  # rating_norm='Round'
        dir_rating_rmse(run, post, epochs, net_type=net_type, weighted=True, configurations=confs)
        #dir_rating_rmse(run, post, epochs, dist='ABS', weighted=True)
        #dir_rating_view(run, post, epochs, net_type=net_type, factor=1)


        #dir_size_rmse(run, post, epochs, net_type=net_type, weighted=False)

        print('=' * 15)
        print('Plots Ready...')
        plt.show()

    finally:

        total_time = (timer() - start) / 60 / 60
        print("Total runtime is {:.1f} hours".format(total_time))
