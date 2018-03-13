import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.manifold import MDS
seed = np.random.RandomState(seed=3)


def symmetrize(a):
    return 0.5*(a + a.T - np.diag(a.diagonal()))


dims        = 4
data_len    = 200
noise_factor = 0.0
data = np.random.randn(data_len, dims)
distance_matrix_base = squareform(pdist(data, 'euclidean'))

sigma = np.std(distance_matrix_base.flatten())
noise_matrix = noise_factor * sigma * np.random.randn(data_len, data_len)
noise_matrix = symmetrize(noise_matrix)
distance_matrix = distance_matrix_base + noise_matrix

weight = np.ones_like(distance_matrix)


# exp 1:    n_components
k = np.zeros(dims)
range_ = range(0, dims)
for d in range_:
    mds = MDS(n_components=(d+1), max_iter=2000, eps=1e-9, random_state=seed, metric=True,
                       dissimilarity="precomputed", n_jobs=1)
    mds_data = mds.fit_transform(distance_matrix, weight=weight)
    mds_distance_matrix = squareform(pdist(mds_data, 'euclidean'))
    k[d] = kendalltau(distance_matrix, mds_distance_matrix)[0]

plt.figure()
plt.plot(np.array(range(dims)) + 1, k, '-*')
plt.xlabel('n_components')
plt.ylabel('kendall correlation')

# exp 2:    noise_factor

noise_factor = [0.0, 0.1, 0.2, 0.4, 0.6, 1.0]
k_noisy = np.zeros(len(noise_factor))
k_orig = np.zeros(len(noise_factor))
for d, nf in enumerate(noise_factor):
    noise_matrix = nf * sigma * np.random.randn(data_len, data_len)
    noise_matrix = symmetrize(noise_matrix)
    distance_matrix = distance_matrix_base + noise_matrix
    mds = MDS(n_components=dims, max_iter=2000, eps=1e-9, random_state=seed, metric=True,
                       dissimilarity="precomputed", n_jobs=1)
    mds_data = mds.fit_transform(distance_matrix, weight=weight)
    mds_distance_matrix = squareform(pdist(mds_data, 'euclidean'))
    k_noisy[d] = kendalltau(distance_matrix, mds_distance_matrix)[0]
    k_orig[d] = kendalltau(distance_matrix_base, mds_distance_matrix)[0]

plt.figure()
plt.plot(noise_factor, k_noisy, '-*')
plt.plot(noise_factor, k_orig, '-*')
plt.xlabel('noise_factor')
plt.ylabel('kendall correlation')
plt.legend(['Noisy', 'Original'])

# exp 3:    n_data

n_data = 100 #dims + 2 + 4
noise_factor = 0.6
noise_matrix = noise_factor * sigma * np.random.randn(data_len, data_len)
noise_matrix = symmetrize(noise_matrix)
distance_matrix = distance_matrix_base + noise_matrix
k_noisy = np.zeros(n_data)
k_orig = np.zeros(n_data)
weight = np.zeros_like(distance_matrix)
for d in range(1, n_data):
    weight[d, :] = 1.0
    weight[:, d] = 1.0
    mds = MDS(n_components=dims, max_iter=2000, eps=1e-9, random_state=seed, metric=True,
                       dissimilarity="precomputed", n_jobs=1)
    mds_data = mds.fit_transform(distance_matrix, weight=weight)
    mds_distance_matrix = squareform(pdist(mds_data, 'euclidean'))
    k_noisy[d] = kendalltau(distance_matrix, mds_distance_matrix)[0]
    k_orig[d] = kendalltau(distance_matrix_base, mds_distance_matrix)[0]

plt.figure()
plt.plot(np.array(range(n_data)), k_noisy, '-*')
plt.plot(np.array(range(n_data)), k_orig, '-*')
plt.xlabel('n_data')
plt.ylabel('kendall correlation')
plt.legend(['Noisy ('+str(noise_factor)+')', 'Original'])

# exp 4:    n_rnd

n_rnd = 20 #dims + 2 + 4
noise_factor = 0.0
noise_matrix = noise_factor * sigma * np.random.randn(data_len, data_len)
noise_matrix = symmetrize(noise_matrix)
distance_matrix = distance_matrix_base + noise_matrix
k_noisy = np.zeros(n_rnd)
k_orig = np.zeros(n_rnd)

for d in range(1, n_rnd):
    p = n_rnd/data_len
    weight = np.floor(np.random.rand(data_len, data_len) / p)
    mds = MDS(n_components=dims, max_iter=2000, eps=1e-9, random_state=seed, metric=True,
                       dissimilarity="precomputed", n_jobs=1)
    mds_data = mds.fit_transform(distance_matrix, weight=weight)
    mds_distance_matrix = squareform(pdist(mds_data, 'euclidean'))
    k_noisy[d] = kendalltau(distance_matrix, mds_distance_matrix)[0]
    k_orig[d] = kendalltau(distance_matrix_base, mds_distance_matrix)[0]

plt.figure()
plt.plot(np.array(range(n_rnd))/data_len, k_noisy, '-*')
plt.plot(np.array(range(n_rnd))/data_len, k_orig, '-*')
plt.xlabel('matrix %')
plt.ylabel('kendall correlation')
plt.legend(['Noisy ('+str(noise_factor)+')', 'Original'])