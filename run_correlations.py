import numpy as np
import matplotlib.pyplot as plt
import pickle

from RatingCorrelator import RatingCorrelator


# ========================
# Setup
# ========================

set = 'Test'
wRuns    = ['000', '001']
nameRuns = ['siam', 'base']

X, Y = 'embed', 'rating'

wEpchs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]

for run, run_name in zip(wRuns, nameRuns):

    WW = ['embed_siam{}-{}_{}.p'.format(run, E, set) for E in wEpchs]

    P, S, K = [], [], []

    for W in WW:
        Reg = RatingCorrelator(W)

        Reg.evaluate_embed_distance_matrix(method='euclidean')

        Reg.evaluate_rating_space()
        Reg.evaluate_rating_distance_matrix(method='euclidean')

        p, s, k = Reg.correlate(X,Y)
        P.append(p)
        S.append(s)
        K.append(k)

    P, S, K = np.array(P), np.array(S), np.array(K)

    #plt.plot(wEpchs, P)
    plt.plot(wEpchs, S)
    plt.plot(wEpchs, K)

plt.title("{}-{}: {} set".format(X,Y,set))
plt.xlabel('Epochs')
plt.ylabel('Correlation')
plt.legend( [   nameRuns[0]+'_Spearman', nameRuns[0]+'Kendall',
                nameRuns[1]+'_Spearman', nameRuns[1]+'Kendall'
             ])

plt.show()