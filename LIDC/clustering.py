import pylidc as pl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, cdist, squareform

ratings = np.array([ann.feature_vals() for ann in pl.query(pl.Annotation).all()])
projection_3d = PCA(n_components=3).fit_transform(ratings)

distance_matrix = squareform(pdist(ratings, 'euclidean'))
affinity_matrix = np.exp(- distance_matrix / distance_matrix.std())

##
# SELECT N OF CLUSTERS
##

n_clusters = np.arange(16, 1025, 16)
n_clusters_scores = list()
for k in n_clusters:
    sc = SpectralClustering(64, affinity='precomputed', assign_labels='kmeans', n_init=100)
    clusters = sc.fit_predict(affinity_matrix)
    cluster_score = list()
    for label in np.unique(clusters):
        cluster_mask = (label == clusters)
        out_of_cluster_mask = np.logical_not(cluster_mask)

        interclass_scores = cdist(projection_3d[cluster_mask, :], projection_3d[out_of_cluster_mask, :], 'euclidean').min(axis=1).mean()
        inclass_scores = pdist(projection_3d[cluster_mask, :], 'euclidean').mean()
        if inclass_scores < 1e-3:
            continue
        score = interclass_scores / inclass_scores
        cluster_score.append(score)

    n_clusters_scores.append(np.array(cluster_score).mean())

plt.figure()
plt.plot(n_clusters, n_clusters_scores, '*-')

##
# BEST
##

n = np.argmax(n_clusters_scores)
sc = SpectralClustering(n_clusters[n], affinity='precomputed', assign_labels='kmeans', n_init=100)
clusters = sc.fit_predict(affinity_matrix)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(projection_3d[:, 0], projection_3d[:, 1], projection_3d[:, 2], c=clusters)

plt.show()
