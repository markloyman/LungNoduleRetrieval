import numpy as np
from scipy.spatial.distance import cdist


def rating_clusters_distance(rating_a, rating_b, distance_norm='eucledean', weight_a=None, weight_b=None):
    dm = cdist(rating_a, rating_b, distance_norm)
    d_b = np.min(dm, axis=0)
    d_a = np.min(dm, axis=1)

    if (weight_a is None) or (weight_b is None):
        distance = 0.5 * np.mean(d_a) + 0.5 * np.mean(d_b)
    else:
        distance = 0.5 * np.dot(d_a, weight_a) / len(d_a) + 0.5 * np.dot(d_b, weight_b) / len(d_b)

    return distance


def rating_clusters_distance_matrix(ratings, distance_norm='euclidean', weights=None):
    n = len(ratings)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i <= j:
                continue
            if weights is None:
                distance = rating_clusters_distance(ratings[i], ratings[j], distance_norm)
            else:
                distance = rating_clusters_distance(ratings[i], ratings[j], distance_norm, weights[i], weights[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix
