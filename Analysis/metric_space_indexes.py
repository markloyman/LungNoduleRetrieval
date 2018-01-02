import numpy as np
from scipy.stats import skew

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
    return index, distribution


def concentration(distance_matrix):
    # Radovanovic, 2010
    dist_std    = np.std( distance_matrix, axis=0)
    dist_mean   = np.mean(distance_matrix, axis=0)
    ratio       = dist_std / dist_mean
    return np.mean(ratio), np.std(ratio), np.mean(dist_std), np.mean(dist_mean)


def relative_contrast(distance_matrix):
    # Aggarwal, On the surprising behavior of distance metrics in high dimensional spaces
    dist_max   = np.max( distance_matrix, axis=0)
    dist_min   = np.min(distance_matrix, axis=0)
    ratio       = (dist_max-dist_min + 1e-6) / (dist_min+1e-6)
    return np.mean(ratio), np.std(ratio)


def relative_contrast_imp(distance_matrix):
    # Aggarwal, On the surprising behavior of distance metrics in high dimensional spaces
    dist_max    = np.max( distance_matrix, axis=0)
    dist_mean   = np.mean(distance_matrix, axis=0)
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
    assert (res > 0)
    end_val = 1.01 * (np.max(nbrs_distances)/sigma - 1)
    range_  = np.arange(-1.0, end_val, res)
    print("kumar resolution: {} - {} samples, sigma={}".format(res, range_.shape[0], sigma))
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Network.data import load_nodule_raw_dataset
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
    metrics = ['l2', 'l2_Scale', 'l2_Norm']
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
        data = load_nodule_raw_dataset(size=144)
        data = data[0] + data[1] + data[2]
    else:
        data = load_nodule_raw_dataset(size=144)[DataSubSet]

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

    rating = []
    for entry in data:
        rating.append(np.mean(entry['rating'], axis=0))
    rating = np.vstack(rating)

    plt.figure()
    plt.subplot(211)
    plt.hist(calc_distance_matrix(rating_normalize(rating, 'Scale'), method='l2').flatten(), bins=500)
    plt.title('L2')
    plt.ylabel('Scaled')
    plt.subplot(212)
    plt.hist(calc_distance_matrix(rating_normalize(rating, 'Norm'), method='l2').flatten(), bins=500)
    #plt.title('l2-norm')
    plt.ylabel('Normalized')



    for metric, m, in zip(metrics, range(len(metrics))):
        plt.figure()
        norm = 'None'
        if len(metric) > 5 and metric[-4:] == 'Norm':
            metric = metric[:-5]
            norm = 'Norm'
        elif len(metric) > 6 and metric[-5:] == 'Scale':
            metric = metric[:-6]
            norm = 'Scale'

        distance_matrix = calc_distance_matrix(rating_normalize(rating, norm), method=metric)
        Ret.fit(len(data)-1, metric=metric, normalization=norm)
        indices, distances = Ret.ret_nbrs()

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
        tau, l_e = kumar(distances, res=0.0001)
        plt.figure()
        plt.plot(l_e[1], l_e[0])
        plt.title('Kummar Tau ({}) = {:.2f}'.format(metric, tau))

        idx_hubness[m] = 1 / np.mean(np.abs(h))
        idx_hubness_std_p[m] = 1 / (np.mean(np.abs(h)) + np.std(np.abs(h)))
        idx_hubness_std_m[m] = 1 / (np.mean(np.abs(h)) - np.std(np.abs(h)))
        idx_symmetry[m] = np.mean(s)
        idx_symmetry_std[m] = np.std(s)
        mean_ratio, std_ratio, std_dist, mean_dist = concentration(distance_matrix)
        idx_concentration[m] = mean_ratio
        idx_concentration_std[m] = std_ratio
        idx_contrast[m] = relative_contrast_imp(distance_matrix)[0]
        idx_contrast_std[m] = relative_contrast_imp(distance_matrix)[1]
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