import numpy as np
from scipy.spatial.distance import cdist


def rating_clusters_distance(rating_a, rating_b, distance_norm='eucledean'):
    dm = cdist(rating_a, rating_b, distance_norm)
    d0 = np.min(dm, axis=0)
    d1 = np.min(dm, axis=1)
    distance = 0.5 * np.mean(d0) + 0.5 * np.mean(d1)
    return distance


def rating_clusters_distance_matrix(ratings, distance_norm='euclidean'):
    n = len(ratings)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i <= j:
                continue
            distance = rating_clusters_distance(ratings[i], ratings[j], distance_norm)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix
