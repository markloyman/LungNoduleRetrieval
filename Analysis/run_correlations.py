import matplotlib.pyplot as plt
import numpy as np

from Analysis.RatingCorrelator import RatingCorrelator

# ========================
# Setup
# ========================
import FileManager
Embed = FileManager.Embed('siam')

dset = 'Valid'
wRuns    = ['064X', '078X']
nameRuns = ['064X', '078X']

X, Y = 'embed', 'rating' #'malig'

metrics = ['l1', 'l2', 'cosine']
wEpchs = [10, 20, 25, 30, 35] #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]

plt.figure()
plt_ = [None]*len(metrics)*2
for i in range(2 * len(metrics)):
    plt_[i] = plt.subplot(len(metrics), 2, i + 1)

for m, metric in enumerate(metrics):
    for run, run_name in zip(wRuns, nameRuns):

        WW = [Embed(run, E, dset) for E in wEpchs]

        P, S, K = [], [], []

        for W in WW:
            Reg = RatingCorrelator(W)

            Reg.evaluate_embed_distance_matrix(method=metric)

            Reg.evaluate_rating_space()
            Reg.evaluate_rating_distance_matrix(method=metric)

            p, s, k = Reg.correlate(X,Y)
            P.append(p)
            S.append(s)
            K.append(k)

        P, S, K = np.array(P), np.array(S), np.array(K)

        #plt.plot(wEpchs, P)
        q = plt_[2 * m + 0].plot(wEpchs, S)
        plt_[2 * m + 1].plot(wEpchs, K, color=q[0].get_color(), ls='--')
        #labels
        plt_[2 * m + 0].axes.yaxis.label.set_text(metric)
        if m == 0: # first row
            plt_[0].axes.title.set_text('Spearman')
            plt_[1].axes.title.set_text('Kendall')
        if m == len(metrics) - 1:  # last row
            plt_[2*m+1].axes.xaxis.label.set_text('epochs')
            plt_[2*m+1].legend(wRuns)

print('Plots Ready...')
plt.show()