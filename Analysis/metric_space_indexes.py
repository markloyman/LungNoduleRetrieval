import numpy as np
from scipy.stats import skew

def k_occurrences(nbrs_indices, k=3):
    nbrs_indices = nbrs_indices[:, :k]
    N = nbrs_indices.shape[0]
    k_occ = np.histogram(nbrs_indices, bins=range(N))
    return k_occ[0]


'''
def skewness(samples):
    u, s = np.mean(samples), np.std(samples)
    skew = (1/s) * np.mean( (samples - u) ** 3)
    return skew
'''


def hubness(nbrs_indices, k=3):
    # Radovanovic, 2010
    k_occ = k_occurrences(nbrs_indices, k)
    index = skew(k_occ)
    return 1.0/index


def concentration(distance_matrix):
    # Radovanovic, 2010
    dist_std    = np.std( distance_matrix, axis=0)
    dist_mean   = np.mean(distance_matrix, axis=0)
    ratio       = dist_std / dist_mean
    return np.mean(ratio), np.std(ratio)


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
    nbrs_indices = nbrs_indices[:, :k]
    N = nbrs_indices.shape[0]
    S = 0
    for ind in range(N):
        for nn in nbrs_indices[ind]:
            if ind in  nbrs_indices[nn]:
                S += 1
    return S / (N*k)


def lambda_p(nbrs_distances, res=0.1):
    lmbd = []
    sigma = np.max(nbrs_distances[:, 0])
    eps  = res*(1 + np.min(nbrs_distances[:, 0]) // res)
    assert (eps > 0)
    n = (1.1 * (np.max(nbrs_distances)/sigma - 1) / eps).astype('int')
    for e in range(n):
        thresh = (e*eps+1)*sigma
        C = np.count_nonzero(nbrs_distances // thresh, axis=1)
        lmbd.append(np.mean(C) / nbrs_distances.shape[0])
    lmbd = np.array(lmbd)
    return lmbd, np.array(eps*range(n))


def kumar(nbrs_distances, res=0.0025):
    l, e = lambda_p(nbrs_distances, res=res)
    tau = res*np.sum(l[np.bitwise_and(l > 0.0001, l < 1)])
    return tau, (l, e)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Network.data import load_nodule_raw_dataset
    from Analysis.retrieval import Retriever
    from Analysis.RatingCorrelator import calc_distance_matrix

    # =======================
    # 0     Test
    # 1     Validation
    # 2     Training
    # =======================
    DataSubSet = 2
    metrics = ['l1', 'l2', 'cosine', 'l1_norm', 'l2_norm', 'cosine_norm']
    #metrics = ['l1', 'l2', 'cosine']

    do_hubness = False
    # =======================

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

    data = load_nodule_raw_dataset(size=144)[DataSubSet]
    Ret = Retriever(title='Rating', dset=post)
    Ret.load_rating(data)

    idx_hubness = np.zeros(len(metrics))
    idx_symmetry = np.zeros(len(metrics))
    idx_concentration = np.zeros(len(metrics))
    idx_concentration_std = np.zeros(len(metrics))
    idx_contrast = np.zeros(len(metrics))
    idx_contrast_std = np.zeros(len(metrics))
    idx_kummar = np.zeros(len(metrics))

    rating = []
    for entry in data:
        rating.append(np.mean(entry['rating'], axis=0))
    rating = np.vstack(rating)

    for metric, m, in zip(metrics, range(len(metrics))):

        distance_matrix = calc_distance_matrix(rating, method=metric)
        norm = 'None'
        if len(metric) > 5 and metric[-4:] == 'norm':
            metric = metric[:-5]
            norm = 'Norm'
        if len(metric) > 5 and metric[-4:] == 'scale':
            metric = metric[:-5]
            norm = 'Scale'
        Ret.fit(len(data)-1, metric=metric, normalization=norm)
        indices, distances = Ret.ret_nbrs()

        #   Hubness
        K = [3, 5, 7, 9, 11, 13]
        h = np.zeros(len(K))
        plt.figure(metric)
        for i in range(len(K)):
            plt.subplot(len(K),1,i+1)
            h[i] = hubness(indices, K[i])
            plt.title('hubness={:.2f}'.format(h[i]))
            plt.hist(k_occurrences(indices, K[i]), bins=10)
            plt.ylabel('k={}'.format(K[i]))

        #   symmetry
        K = [3, 5, 7, 9, 11, 13]
        s = np.zeros(len(K))
        for i in range(len(K)):
            s[i] = symmetry(indices, K[i])
        plt.figure('sym: '+metric)
        plt.title('symmetry={:.2f}'.format(np.mean(s)))
        plt.plot(K, s)
        plt.ylabel('symmetry')
        plt.xlabel('K')

        # kumar index
        tau, l_e = kumar(distances, res=0.001)
        plt.figure()
        plt.plot(l_e[1], l_e[0])
        plt.title('Kummar Tau ({}) = {:.2f}'.format(metric, tau))

        idx_hubness[m] = np.mean(h)
        idx_symmetry[m] = np.mean(s)
        idx_concentration[m] = concentration(distance_matrix)[0]
        idx_concentration_std[m] = concentration(distance_matrix)[1]
        idx_contrast[m] = relative_contrast_imp(distance_matrix)[0]
        idx_contrast_std[m] = relative_contrast_imp(distance_matrix)[1]
        idx_kummar[m] = tau

        print("{} Metric:".format(metric))
        print('-' * 15)
        print("Hubness = {:.2f}".format(idx_hubness[m]))
        print("Concentration = {:.2f} (std {:.2f})".format(idx_concentration[m], idx_concentration_std[m]))
        print("*Contrast* = {:.2f} (std {:.2f})".format(idx_contrast[m], idx_contrast_std[m]))
        print("Symmetry = {:.2f}".format(idx_symmetry[m]))
        print("Kumamr Index = {:.2f}".format(tau))

    plt.figure()
    x = np.array(range(len(metrics)))
    plt.subplot(511)
    plt.grid(which='major', axis='y')
    plt.bar(x, idx_hubness, align='center', alpha=0.5)
    plt.xticks(x, metrics)
    plt.ylabel('Hubness')
    plt.subplot(512)
    plt.grid(which='major', axis='y')
    plt.bar(x, idx_concentration, align='center', alpha=0.5)
    plt.xticks(x, metrics)
    plt.ylabel('Concentration')
    plt.subplot(513)
    plt.grid(which='major', axis='y')
    plt.bar(x, idx_contrast, align='center', alpha=0.5)
    plt.xticks(x, metrics)
    plt.ylabel('Contrast')
    plt.subplot(514)
    plt.grid(which='major', axis='y')
    plt.bar(x, idx_symmetry, align='center', alpha=0.5)
    plt.xticks(x, metrics)
    plt.ylabel('Symmetry')
    plt.subplot(515)
    plt.grid(which='major', axis='y')
    plt.bar(x, idx_kummar, align='center', alpha=0.5)
    plt.xticks(x, metrics)
    plt.ylabel('Kumamr')

    plt.show()