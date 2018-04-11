from init import *
from Analysis import metric_space_indexes as index
from Analysis.RatingCorrelator import calc_distance_matrix
from Analysis.retrieval import Retriever
from Network import FileManager
from experiments import load_experiments
dset = 'Valid'
rating_normalizaion = 'Scale' # 'None', 'Normal', 'Scale'
metrics = ['l2']
n_groups = 5


runs, run_net_types, run_metrics, run_epochs, run_names = load_experiments('Pooling')


plt.figure("Metric Space - {}".format(dset))
p = [None] * len(metrics) * 5
for i in range(5 * len(metrics)):
    p[i] = plt.subplot(len(metrics), 5, i + 1)
for m, metric in enumerate(metrics):
    print("Begin: {} metric".format(metric))
    for run, net_type, r, epochs in zip(runs, run_net_types, range(len(runs)), run_epochs):
        Embed = FileManager.Embed(net_type)
        embed_source = [Embed(run + 'c{}'.format(c), dset) for c in range(n_groups)]
        #WW = [Embed(run, dset) for E in epochs]
        # init
        x_len = len(epochs)
        idx_hubness = np.zeros(x_len)
        idx_hubness_std = np.zeros(x_len)
        idx_symmetry = np.zeros(x_len)
        idx_symmetry_std = np.zeros(x_len)
        idx_concentration = np.zeros(x_len)
        idx_concentration_std = np.zeros(x_len)
        idx_contrast = np.zeros(x_len)
        idx_contrast_std = np.zeros(x_len)
        idx_kummar = np.zeros(x_len)
        # calculate
        Ret = Retriever(title='{}'.format(run), dset=dset)
        embd, epoch_mask = Ret.load_embedding(embed_source, multi_epcch=True)
        for e, E in enumerate(epochs):
            Ret.fit(metric=metric, epoch=E)
            indices, distances = Ret.ret_nbrs()
            distance_matrix = calc_distance_matrix(embd[np.argwhere(E == epoch_mask)[0][0]], metric)
            # hubness
            K = [5, 7]  # [3, 5, 7, 11, 17]
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
            K = [5, 7]  # [3, 5, 7, 11, 17]
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