from init import *
from Analysis.retrieval import Retriever
from Analysis import metric_space_indexes as index
from Analysis.RatingCorrelator import calc_distance_matrix


dset = 'Valid'
rating_normalizaion = 'Scale' # 'None', 'Normal', 'Scale'
metrics = ['l2']

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
run_epochs      = [ [5, 15, 25, 35, 45, 55],
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
#wRuns = ['011X', '016XXXX', '023X']  #['064X', '078X', '026'] #['064X', '071' (is actually 071X), '078X', '081', '082']
#wRunsNet = ['dirR', 'trip', 'trip']  #, 'dir']
#run_metrics = ['l2', 'l2', 'l2']

#wRuns            = ['021', '022XX', '023X', '025']
#run_names       = ['max-pool', 'rmac', 'categ', 'confidence+cat' ]

wRuns            = ['011XXX', '016XXXX', '025']
run_names       = ['malig', 'rating', 'pretrain' ]

wRunsNet   = ['trip']*len(wRuns)
run_metrics     = ['l2']*len(wRuns)

#wEpchs = [[15, 35, 55, 95],
#          [20, 40, 100],
#          [5, 15, 25, 35]]

wEpchs = [  [5, 15, 25, 35, 45, 55],
            [20, 40, 100, 150, 180],
            [5, 15, 25, 35, 45, 55]
        ]

leg = ['E{}'.format(E) for E in wEpchs]
'''


plt.figure("Metric Space - {}".format(dset))
p = [None] * len(metrics) * 5
for i in range(5 * len(metrics)):
    p[i] = plt.subplot(len(metrics), 5, i + 1)
for m, metric in enumerate(metrics):
    print("Begin: {} metric".format(metric))
    for run, net_type, r, epochs in zip(runs, run_net_types, range(len(runs)), run_epochs):
        Embed = FileManager.Embed(net_type)
        WW = [Embed(run, E, dset) for E in epochs]
        # init
        idx_hubness = np.zeros(len(WW))
        idx_hubness_std = np.zeros(len(WW))
        idx_symmetry = np.zeros(len(WW))
        idx_symmetry_std = np.zeros(len(WW))
        idx_concentration = np.zeros(len(WW))
        idx_concentration_std = np.zeros(len(WW))
        idx_contrast = np.zeros(len(WW))
        idx_contrast_std = np.zeros(len(WW))
        idx_kummar = np.zeros(len(WW))
        # calculate
        for e, W in enumerate(WW):
            Ret = Retriever(title='{}'.format(run), dset=dset)
            embd = Ret.load_embedding(W)
            Ret.fit(metric=metric)
            indices, distances = Ret.ret_nbrs()
            distance_matrix = calc_distance_matrix(embd, metric)
            # hubness
            K = [3, 5, 7, 11, 17]
            h = np.zeros(len(K))
            # plt.figure()
            for i in range(len(K)):
                h_ = index.hubness(indices, K[i])
                h[i] = h_[0]
                if K[i] == 3:
                    j = 1
                elif K[i] == 7:
                    j = 2
                elif K[i] == 11:
                    j = 3
                else:
                    j = 0
                if False:  # j != 0:
                    plt.subplot(len(metrics), 3, m * 3 + j)
                    plt.title('k-occ: {}'.format(net_type))
                    plt.ylabel('k={}'.format(K[i]))
                    plt.plot(np.array(range(len(h_[1]))), h_[1])
            idx_hubness[e] = np.mean(h)
            idx_hubness_std[e] = np.std(h)
            #   symmetry
            K = [3, 5, 7, 11, 17]
            s = np.zeros(len(K))
            for i in range(len(K)):
                s[i] = index.symmetry(indices, K[i])
            idx_symmetry[e] = np.mean(s)
            idx_symmetry_std[e] = np.std(s)
            # kumar index
            tau, l_e = index.kumar(distances, res=0.0001)
            idx_kummar[e] = tau
            idx_concentration[e] = index.concentration(distance_matrix)[0]
            idx_concentration_std[e] = index.concentration(distance_matrix)[1]
            idx_contrast[e] = index.relative_contrast_imp(distance_matrix)[0]
            idx_contrast_std[e] = index.relative_contrast_imp(distance_matrix)[1]
        # plot
        #   hubness
        q = p[5 * m + 0].plot(epochs, idx_hubness)
        p[5 * m + 0].plot(epochs, idx_hubness + idx_hubness_std, color=q[0].get_color(), ls='--')
        p[5 * m + 0].plot(epochs, idx_hubness - idx_hubness_std, color=q[0].get_color(), ls='--')
        #   symmetry
        q = p[5 * m + 1].plot(epochs, idx_symmetry)
        p[5 * m + 1].plot(epochs, idx_symmetry + idx_symmetry_std, color=q[0].get_color(), ls='--')
        p[5 * m + 1].plot(epochs, idx_symmetry - idx_symmetry_std, color=q[0].get_color(), ls='--')
        #   contrast
        q = p[5 * m + 2].plot(epochs, idx_contrast)
        p[5 * m + 2].plot(epochs, idx_contrast + idx_contrast_std, color=q[0].get_color(), ls='--')
        p[5 * m + 2].plot(epochs, idx_contrast - idx_contrast_std, color=q[0].get_color(), ls='--')
        #   concentration
        q = p[5 * m + 3].plot(epochs, idx_concentration)
        p[5 * m + 3].plot(epochs, idx_concentration + idx_concentration_std, color=q[0].get_color(), ls='--')
        p[5 * m + 3].plot(epochs, idx_concentration - idx_concentration_std, color=q[0].get_color(), ls='--')
        #   kumar
        p[5 * m + 4].plot(epochs, idx_kummar)
        # labels
        if r == 0:  # first column
            p[5 * m + 0].axes.yaxis.label.set_text(metric)
        if m == 0:  # first row
            p[5 * m + 0].axes.title.set_text('hubness')
            p[5 * m + 1].axes.title.set_text('symmetry')
            p[5 * m + 2].axes.title.set_text('contrast')
            p[5 * m + 3].axes.title.set_text('concentration')
            p[5 * m + 4].axes.title.set_text('kumari')
        if m == len(metrics) - 1:  # last row
            p[5 * m + 2].axes.xaxis.label.set_text('epochs')
p[-1].legend(run_names)
print('Done doMetricSpaceIndexes')

plt.show()