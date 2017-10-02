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
    return index


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