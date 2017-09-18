import numpy as np
import pickle
import matplotlib.pyplot as plt

from LIDC.lidcUtils import getAnnotation

def maligancy_label_vs_maligancy_rating(labels, meta_data):
    malig = np.array([getAnnotation(meta).feature_vals()[-1] for meta in meta_data]).reshape(-1, 1)

    plt.figure()
    plt.plot( [1,2,3,4,5], np.histogram(malig[labels == 0], 5)[0])
    plt.plot( [1,2,3,4,5], np.histogram(malig[labels == 1], 5)[0])
    plt.xlabel('Maligancy Rating')
    plt.ylabel('Count')
    plt.legend(['Benign', 'Malignant'])

def hist(x, y, nx, ny):
    H = np.histogram2d(np.squeeze(x), np.squeeze(y), [nx, ny])[0].T
    plt.plot(H)
