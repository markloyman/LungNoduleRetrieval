from init import *
from Analysis import RatingCorrelator

# ========================
# Setup
# ========================

dset = 'Valid'
X, Y = 'embed', 'rating' #'malig' 'rating'
metrics = ['l2']  #['l2', 'l1', 'cosine']
rating_norm = 'Normal'

'''
# ===========================
#   Malignancy Objective
# ===========================
runs            = ['103', '100', '011XXX']
run_net_types   = ['dir', 'siam', 'trip']
run_metrics     = ['l2']*len(runs)
run_epochs      = [ [5, 10, 15, 20, 25, 30, 35, 40, 45],
                    [5, 10, 15, 20, 25, 30, 35, 40, 45],
                    [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
                ]
run_names       = run_net_types
# ===========================
'''

'''
# ===========================
#   Triplet Compare to dirR reference
# ===========================
runs            = ['011X', '011XXX', '016XXXX', '023X']
run_net_types   = ['dirR', 'trip', 'trip', 'trip']
run_metrics     = ['l2']*len(runs)
run_epochs      = [ [5, 15, 25, 35, 45, 55, 65, 75, 85],
                    [5, 15, 25, 35, 45, 55],
                    [20, 40, 100, 150, 180],
                    [5, 15, 20, 25, 30, 35]
                ]
run_names       = ['dirR', 'malig-obj', 'trip', 'trip-finetuned']
# ===========================
'''

#'''
# ===========================
#   Triplets
# ===========================
runs            = ['011XXX', '016XXXX', '027', '023X']
run_net_types   = ['trip']*len(runs)
run_metrics     = ['l2']*len(runs)
run_epochs      = [ [5, 15, 25, 35, 45, 55],
                    [20, 40, 100, 150, 180],
                    [5, 15, 25, 35, 40, 45, 50, 55, 60],
                    [5, 15, 20, 25, 30, 35]
                ]
run_names       = ['malig-obj', 'rating-obj', 'rating-obj', 'trip-finetuned']
# ===========================
#'''

'''    
#wRuns   = ['011X', '016XXXX', '021', '023X']  #['100', '101', '103', '103']
#wRunNet = ['dirR', 'trip', 'trip', 'trip']   #['siam', 'siam', 'siam', 'dir']
#wDist   = ['l2', 'l2', 'l2', 'l2']         #['l2', 'l1', 'cosine', 'l2']

wRuns            = ['021', '022XX', '023X', '025']
run_names       = ['max-pool', 'rmac', 'categ', 'confidence+cat' ]
wRunNet   = ['trip']*len(wRuns)
wDist     = ['l2']*len(wRuns)



wEpchs = [ [5, 15, 25, 35],
            [5, 15, 25, 35, 45, 55],
            [5, 15, 25, 35],
            [5, 15, 25, 35, 45, 55]
        ]
'''

plt.figure()
plt_ = [None]*len(metrics)*2
for i in range(2 * len(metrics)):
    plt_[i] = plt.subplot(len(metrics), 2, i + 1)

for m, metric in enumerate(metrics):
    for run, net_type, dist, epochs in zip(runs, run_net_types, run_metrics, run_epochs):
        Embed = FileManager.Embed(net_type)
        WW = [Embed(run, E, dset) for E in epochs]

        # correlation plot
        P, S, K = [], [], []
        for W in WW:
            Reg = RatingCorrelator(W)
            Reg.evaluate_embed_distance_matrix(method=dist)
            Reg.evaluate_rating_space(norm=rating_norm)
            Reg.evaluate_rating_distance_matrix(method=metric)

            p, s, k = Reg.correlate_retrieval(X, Y)
            #p, s, k = Reg.correlate(X, Y)
            P.append(p)
            S.append(s)
            K.append(k)
        P, S, K = np.array(P), np.array(S), np.array(K)

        # correlation distribution
        #for W in WW[-1:]:
        #    Reg = RatingCorrelator(W)
        #    Reg.evaluate_embed_distance_matrix(method=dist)
        #    Reg.evaluate_rating_space(norm=rating_norm)
        #    Reg.evaluate_rating_distance_matrix(method=metric)
        #    K_hist_x, K_hist_y = Reg.kendall_histogram(X, Y)


        #plt.plot(wEpchs, P)
        #plt_[2 * m + 0].plot(K_hist_x, K_hist_y)
        plt_[2 * m + 0].plot(epochs, P)
        plt_[2 * m + 0].grid(which='major', axis='y')
        plt_[2 * m + 1].plot(epochs, K)
        plt_[2 * m + 1].grid(which='major', axis='y')
        #labels
        plt_[2 * m + 0].axes.yaxis.label.set_text(metric)
        if m == 0: # first row
            plt_[0].axes.title.set_text('Pearson')
            plt_[1].axes.title.set_text('Kendall')
        if m == len(metrics) - 1:  # last row
            plt_[2*m+1].axes.xaxis.label.set_text('epochs')
            plt_[2*m+1].legend(run_names) #[n+r for n,r in zip(wRunNet, wRuns)])

#Embed = FileManager.Embed('siamR')
#Reg = RatingCorrelator(Embed('006XX', 40, dset))
#Reg.evaluate_embed_distance_matrix(method='l2')
#Reg.evaluate_rating_space(norm=rating_norm)
#Reg.evaluate_rating_distance_matrix(method='l2')
#Reg.scatter('embed', 'rating', xMethod="euclidean", yMethod='euclidean', sub=False)

print('Plots Ready...')
plt.show()