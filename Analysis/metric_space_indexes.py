import numpy as np
from scipy.stats import skew
from Analysis.performance import mean_cross_validated_index
from Analysis.retrieval import Retriever
from Network import FileManager



def k_occurrences(nbrs_indices, k=3):
    nbrs_indices_ = nbrs_indices[:, :k]
    k_occ = np.bincount(nbrs_indices_.flatten())
    return k_occ


'''
def skewness(samples):
    u, s = np.mean(samples), np.std(samples)
    skew = (1/s) * np.mean( (samples - u) ** 3)
    return skew
'''


def hubness(nbrs_indices, k=3):
    # Radovanovic, 2010
    k_occ = k_occurrences(nbrs_indices, k)
    distribution = np.bincount(k_occ)
    index = skew(distribution)
    #index = 1 / (1e-6 + index)
    index = np.exp(-np.abs(index))
    return index, distribution


def concentration(distance_matrix):
    # Radovanovic, 2010
    dist_std    = np.std( distance_matrix, axis=1)
    dist_mean   = np.mean(distance_matrix, axis=1)
    ratio       = dist_std / dist_mean
    return np.mean(ratio), np.std(ratio), np.mean(dist_std), np.mean(dist_mean)


def relative_contrast(distance_matrix):
    # Aggarwal, On the surprising behavior of distance metrics in high dimensional spaces
    dist_max   = np.max( distance_matrix, axis=1)
    dist_min   = np.min(distance_matrix, axis=1)
    ratio       = (dist_max-dist_min - 1e-6) / (dist_min+1e-6)
    return np.mean(ratio), np.std(ratio)


def relative_contrast_imp(distance_matrix):
    # Aggarwal, On the surprising behavior of distance metrics in high dimensional spaces
    dist_max    = np.max( distance_matrix, axis=1)
    dist_mean   = np.mean(distance_matrix, axis=1)
    ratio       = (dist_max-dist_mean) / dist_mean
    return np.mean(ratio), np.std(ratio)


def symmetry(nbrs_indices, k=3):
    nbrs_indices_ = nbrs_indices[:, :k]
    N = nbrs_indices_.shape[0]
    S = 0
    for ind in range(N):
        for nn in nbrs_indices_[ind]:
            if ind in  nbrs_indices_[nn]:
                S += 1
    return S / (N*k)


def lambda_p(nbrs_distances, res=0.1):
    lmbd = []
    sigma = np.mean(nbrs_distances[:, 0])
    assert sigma > 0
    assert res > 0
    end_val = 1.01 * (np.max(nbrs_distances)/sigma - 1)
    range_  = np.arange(-1.0, end_val, res)
    print("\tkumar resolution: {} - {} samples, sigma={:.2f}".format(res, range_.shape[0], sigma))
    for eps in range_:
        thresh = (eps+1)*sigma
        C = np.count_nonzero(nbrs_distances > thresh, axis=1)
        lmbd.append(np.mean(C) / nbrs_distances.shape[0])
    lmbd = np.array(lmbd)
    return lmbd, range_


def kumar(nbrs_distances, res=0.0025):
    l, e = lambda_p(nbrs_distances, res=res)
    tau = res*np.sum(l[np.bitwise_and(l > 0, l < 1)])
    return tau, (l, e)


def calc_hubness(indices, K=[3, 5, 7, 11, 17], verbose=False):
    if verbose:
        plt.figure()
    h = np.zeros(len(K))
    for i in range(len(K)):
        h_ = hubness(indices, K[i])
        h[i] = h_[0]
        if verbose:
            #plt.subplot(1,len(K), i+1)
            plt.plot(h_[1])
    h_mean, h_std = np.mean(h), np.std(h)
    if verbose:
        plt.legend(['{}: {:.2f}'.format(k, ind) for k, ind in zip(K, h)])
        plt.title('Hubness = {:.2f} ({:.2f})'.format(h_mean, h_std))
    return h_mean, h_std


def calc_symmetry(indices, K=[3, 5, 7, 11, 17]):
    s = np.zeros(len(K))
    for i in range(len(K)):
        s[i] = symmetry(indices, K[i])
    return np.mean(s), np.std(s)


def distances_distribution(distances):
    plt.figure()
    for dist in distances:
        hist, bins = np.histogram(dist, bins=20)
        axis = bins[:-1] + 0.5*(bins[1:] - bins[:-1])
        plt.plot(axis, hist, marker='*', alpha=0.2)


def features_correlation(embedding):
    # normalize data
    embedding = embedding - np.mean(embedding, axis=0, keepdims=True)
    embedding /= np.std(embedding, axis=0, keepdims=True) + 1e-9
    # covariance (correlation) matrix
    cov = embedding.transpose().dot(embedding)
    cov /= embedding.shape[0]
    # get off-diagonal
    mask = np.ones_like(cov) - np.eye(embedding.shape[1])
    vals = (cov*mask).flatten()
    l1 = np.mean(np.abs(vals))
    return l1


def samples_correlation(embedding):
    # normalize data
    embedding = embedding - np.mean(embedding, axis=1, keepdims=True)
    embedding /= np.std(embedding, axis=1, keepdims=True) + 1e-9
    # covariance (correlation) matrix
    cov = embedding.dot(embedding.transpose())
    cov /= embedding.shape[1]
    # get off-diagonal
    mask = np.ones_like(cov) - np.eye(embedding.shape[0])
    vals = (cov*mask).flatten()
    l1 = np.mean(np.abs(vals))
    return l1


def eval_embed_space(run, net_type, metric, rating_metric, epochs, dset, rating_norm='none', cross_validation=False, n_groups=5):
    # init
    Embed = FileManager.Embed(net_type)
    embed_source = [Embed(run + 'c{}'.format(c), dset) for c in range(n_groups)]
    idx_hubness, idx_symmetry, idx_concentration, idx_contrast, idx_kummar, idx_featCorr, idx_sampCorr \
        = [[] for i in range(n_groups)], [[] for i in range(n_groups)], [[] for i in range(n_groups)], \
          [[] for i in range(n_groups)], [[] for i in range(n_groups)], [[] for i in range(n_groups)], \
          [[] for i in range(n_groups)]
    valid_epochs = [[] for i in range(n_groups)]
    # calculate
    Ret = Retriever(title='{}'.format(run), dset=dset)
    for i, source in enumerate(embed_source):
        embd, epoch_mask = Ret.load_embedding(source, multi_epcch=True)
        for e in epochs:
            try:
                epoch_idx = np.argwhere(e == epoch_mask)[0][0]
                Ret.fit(metric=metric, epoch=e)
                indices, distances = Ret.ret_nbrs()
                # hubness
                idx_hubness[i].append(calc_hubness(indices))
                #   symmetry
                idx_symmetry[i].append(calc_symmetry(indices))
                # kumar index
                tau, l_e = kumar(distances, res=0.01)
                idx_kummar[i].append(tau)
                # concentration & contrast
                idx_concentration[i].append(concentration(distances))
                idx_contrast[i].append(relative_contrast_imp(distances))
                valid_epochs[i].append(e)
                # correlation
                idx_featCorr[i].append(features_correlation(embd[epoch_idx]))
                idx_sampCorr[i].append(samples_correlation(embd[epoch_idx]))
            except:
                print("Epoch {} - no calculated embedding".format(e))
        valid_epochs[i] = np.array(valid_epochs[i])
        idx_hubness[i] = np.array(list(zip(*idx_hubness[i])))
        idx_symmetry[i] = np.array(list(zip(*idx_symmetry[i])))
        idx_concentration[i] = np.array(list(zip(*idx_concentration[i])))
        idx_contrast[i] = np.array(list(zip(*idx_contrast[i])))
        idx_kummar[i] = np.array([idx_kummar[i]])
        idx_featCorr[i] = np.array([idx_featCorr[i]])
        idx_sampCorr[i] = np.array([idx_sampCorr[i]])

    combined_epochs = [i for i, c in enumerate(np.bincount(np.concatenate(valid_epochs))) if c > 3]

    idx_hubness = mean_cross_validated_index(idx_hubness, valid_epochs, combined_epochs)
    idx_symmetry = mean_cross_validated_index(idx_symmetry, valid_epochs, combined_epochs)
    idx_concentration = mean_cross_validated_index(idx_concentration, valid_epochs, combined_epochs)
    idx_contrast = mean_cross_validated_index(idx_contrast, valid_epochs, combined_epochs)
    idx_kummar = mean_cross_validated_index(idx_kummar, valid_epochs, combined_epochs)
    idx_featCorr = mean_cross_validated_index(idx_featCorr, valid_epochs, combined_epochs)
    idx_sampCorr = mean_cross_validated_index(idx_sampCorr, valid_epochs, combined_epochs)

    return combined_epochs, idx_hubness, idx_symmetry, idx_concentration, idx_contrast, idx_kummar, idx_featCorr, idx_sampCorr


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Network.data_loader import load_nodule_dataset
    from Analysis.retrieval import Retriever
    from Analysis.RatingCorrelator import calc_distance_matrix
    from Network.dataUtils import rating_normalize

    # =======================
    # 0     Test
    # 1     Validation
    # 2     Training
    # =======================
    DataSubSet = -1
    #metrics = ['l1_Scale', 'l1_Norm', 'l2_Scale', 'l2_Norm', 'cosine_Scale', 'cosine_Norm', 'correlation_Scale', 'correlation_Norm']
    metrics = ['l2']  # ['l2', 'l2_Scale', 'l2_Norm']
    #

    do_hubness = False
    # =======================

    if DataSubSet == 0:
        post = "Test"
    elif DataSubSet == 1:
        post = "Valid"
    elif DataSubSet == 2:
        post = "Train"
    elif DataSubSet == -1:
        post = "All"
    else:
        assert False
    print("{} Set Analysis".format(post))
    print('=' * 15)

    if DataSubSet == -1:
        data = load_nodule_dataset(size=160, sample='Normal', res=0.5, configuration=0)
        data = data[0] + data[1] + data[2]
    else:
        data = load_nodule_dataset(size=160)[DataSubSet]

    Ret = Retriever(title='Rating', dset=post)
    Ret.load_rating(data)

    idx_hubness = np.zeros(len(metrics))
    idx_hubness_std_p = np.zeros(len(metrics))
    idx_hubness_std_m = np.zeros(len(metrics))
    idx_symmetry = np.zeros(len(metrics))
    idx_symmetry_std = np.zeros(len(metrics))
    idx_concentration = np.zeros(len(metrics))
    idx_concentration_std = np.zeros(len(metrics))
    idx_contrast = np.zeros(len(metrics))
    idx_contrast_std = np.zeros(len(metrics))
    idx_kummar = np.zeros(len(metrics))

    #rating = []
    #for entry in data:
    #    rating.append(np.mean(entry['rating'], axis=0))
    #rating = np.vstack(rating)

    #plt.figure()
    #plt.subplot(311)
    #plt.hist(calc_distance_matrix(rating_normalize(rating, 'none'), method='l2').flatten(), bins=500)
    #plt.title('L2')
    #plt.ylabel('none')
    #plt.subplot(312)
    #plt.hist(calc_distance_matrix(rating_normalize(rating, 'Scale'), method='l2').flatten(), bins=500)
    #plt.title('l2-norm')
    #plt.ylabel('Scale')
    #plt.subplot(313)
    #plt.hist(calc_distance_matrix(rating_normalize(rating, 'Norm'), method='l2').flatten(), bins=500)
    # plt.title('l2-norm')
    #plt.ylabel('Normalized')

    #plt.show()

    for metric, m, in zip(metrics, range(len(metrics))):

        norm = 'None'
        if len(metric) > 5 and metric[-4:] == 'Norm':
            metric = metric[:-5]
            norm = 'Norm'
        elif len(metric) > 6 and metric[-5:] == 'Scale':
            metric = metric[:-6]
            norm = 'Scale'

        #distance_matrix = calc_distance_matrix(rating_normalize(rating, norm), method=metric)
        Ret.fit(len(data)-1, metric=metric, normalization=norm)
        indices, distances = Ret.ret_nbrs()

        plt.figure()
        plt.hist(distances.flatten(), bins=500)
        plt.title('distance distribution')

        plt.figure()

        #   Hubness
        K = [3, 5, 7, 11, 17]
        #K = 1+np.array(range(20))
        h = np.zeros(len(K))

        plt.figure('k_occ')
        plt.title('k_occ distribution')
        for i in range(len(K)):
            #plt.subplot(len(metrics),1,m+1)
            #plt.subplot(len(K), 1, i + 1)
            h_ = hubness(indices, K[i])
            h[i] = h_[0]
            if metric == 'correlation' or metric == 'l2':
                if K[i]==3:
                    j = 1
                elif K[i]==7:
                    j = 2
                elif K[i]==11:
                    j = 3
                else:
                    j = 0
                if j !=0:
                    plt.title('{}-{}'.format(metric, norm))
                    plt.ylabel('k={}'.format(K[i]))
                    plt.subplot(len(metrics), 3, m*3 + j)
                    plt.plot(np.array(range(len(h_[1]))), h_[1])
            #plt.title('hubness={:.2f}'.format(h[i]))
            #plt.hist(k_occurrences(indices, K[i]), bins=10)
            #H = np.histogram(k_occurrences(indices, K[i]))
            #plt.plot(0.5*(H[1][1:] + H[1][:-1]), H[0])
            #plt.plot(np.array(range(len(h_[1]))), h_[1])
            #plt.ylabel(metric)
            #plt.xlabel('H: {}'.format(h_[0]))
            #plt.ylabel('k={}'.format(K[i]))
        plt.figure('Hubness')
        plt.subplot(len(metrics), 1, m + 1)
        plt.plot(K, h)
        plt.xlabel('K')
        plt.ylabel(metric+'-'+norm if norm != 'None' else '')

        #   symmetry
        K = [3, 5, 7, 11, 17]
        #K = 1 + np.array(range(20))
        s = np.zeros(len(K))
        for i in range(len(K)):
            s[i] = symmetry(indices, K[i])
        plt.figure()
        plt.title('{} symmetry={:.2f}'.format(metric, np.mean(s)))
        plt.plot(K, s)
        plt.ylabel('symmetry')
        plt.xlabel('epoch')

        # kumar index
        tau = 0
        tau, l_e = kumar(distances, res=0.01)
        plt.figure()
        plt.plot(l_e[1], l_e[0])
        plt.title('Kummar Tau ({}) = {:.2f}'.format(metric, tau))

        idx_hubness[m] = np.mean(np.abs(h))
        idx_hubness_std_p[m] = (np.mean(np.abs(h)) + np.std(np.abs(h)))
        idx_hubness_std_m[m] = (np.mean(np.abs(h)) - np.std(np.abs(h)))
        idx_symmetry[m] = np.mean(s)
        idx_symmetry_std[m] = np.std(s)
        mean_ratio, std_ratio, std_dist, mean_dist = concentration(distances)
        idx_concentration[m] = mean_ratio
        idx_concentration_std[m] = std_ratio
        idx_contrast[m] = relative_contrast_imp(distances)[0]
        idx_contrast_std[m] = relative_contrast_imp(distances)[1]
        idx_kummar[m] = tau

        print("\n{} Metric:".format(metric))
        print('-' * 15)
        print("Hubness = {:.2f} (std {:.2f}-{:.2f})".format(idx_hubness[m], idx_hubness_std_m[m], idx_hubness_std_p[m]))
        print("Symmetry = {:.2f} (std {:.2f})".format(idx_symmetry[m], idx_symmetry_std[m]))
        print("Concentration = {:.2f} (std {:.2f}), std_dist: {}, mean_dist: {})".format(idx_concentration[m], idx_concentration_std[m],std_dist, mean_dist))
        print("*Contrast* = {:.2f} (std {:.2f}".format(idx_contrast[m], idx_contrast_std[m]))
        print("Kumamr Index = {:.2f}".format(tau))

    plt.figure()
    x = np.array(range(len(metrics)))
    plt.subplot(511)
    plt.grid(which='major', axis='y')
    plt.bar(x, idx_hubness, align='center', alpha=0.5)
    plt.xticks(x, metrics)
    plt.ylabel('Hubness')
    plt.subplot(513)
    plt.grid(which='major', axis='y')
    plt.bar(x, idx_concentration, align='center', alpha=0.5)
    plt.xticks(x, metrics)
    plt.ylabel('Concentration')
    plt.subplot(514)
    plt.grid(which='major', axis='y')
    plt.bar(x, idx_contrast, align='center', alpha=0.5)
    plt.xticks(x, metrics)
    plt.ylabel('Contrast')
    plt.subplot(512)
    plt.grid(which='major', axis='y')
    plt.bar(x, idx_symmetry, align='center', alpha=0.5)
    plt.xticks(x, metrics)
    plt.ylabel('Symmetry')
    plt.subplot(515)
    plt.grid(which='major', axis='y')
    plt.bar(x, idx_kummar, align='center', alpha=0.5)
    plt.xticks(x, metrics)
    plt.ylabel('Kumamr')

    print('DONE')
    plt.show()