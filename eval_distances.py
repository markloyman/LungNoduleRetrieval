from init import *
from Analysis import metric_space_indexes as index
from Analysis.RatingCorrelator import calc_distance_matrix
from Analysis.retrieval import Retriever
from Network import FileManager
from experiments import load_experiments
from scipy.signal import savgol_filter
dset = 'Valid'
rating_normalizaion = 'Scale' # 'None', 'Normal', 'Scale'
metrics = ['l2']
n_groups = 5


def plot_row(i, distances, label=None):
    tau, l_e = index.kumar(distances, res=0.01)
    conc = index.concentration(distances)
    contrast = index.relative_contrast_imp(distances)

    p[i].hist(distances.flatten() / np.mean(distances), bins=20)
    p[i].axes.title.set_text('std:{:.2f}, conc:{:.2f}, ctrst:{:.2f}'.format(conc[2], conc[0], contrast[0]))
    p[i].axes.yaxis.label.set_text('distribution')
    p[i].axes.xaxis.label.set_text('distances')
    Ret.pca(epoch=e, plt_=p[i + 1], label=label)
    p[i + 2].plot(l_e[1], l_e[0])
    p[i + 2].axes.title.set_text('Kumari (tau = {:.2f}'.format(tau))


runs, run_net_types, run_metrics, run_epochs, run_names, _, _ = load_experiments('SiamRating')


# evaluate

Epochs, Idx_hubness, Idx_symmetry, Idx_concentration, Idx_contrast, Idx_kummar = [], [], [], [], [], []

for m, metric in enumerate(metrics):
    print("Begin: {} metric".format(metric))
    for run, net_type, r, epochs, name in zip(runs, run_net_types, range(len(runs)), run_epochs, run_names):
        print("Evaluating run {}{}".format(net_type, run))
        # initialize figures
        plt.figure("Distances - {}".format(name))
        p = [None] * 9
        for i in range(9):
            p[i] = plt.subplot(3, 3, i + 1)
        # init
        Embed = FileManager.Embed(net_type)
        embed_source = [Embed(run + 'c{}'.format(c), dset) for c in range(n_groups)]
        idx_hubness, idx_symmetry, idx_concentration, idx_contrast, idx_kummar, valid_epochs = [], [], [], [], [], []
        # calculate
        Ret = Retriever(title='{}'.format(run), dset=dset)
        embd, epoch_mask = Ret.load_embedding(embed_source, multi_epcch=True)
        for e in [60]: # epochs:
            # full
            Ret.fit(metric=metric, epoch=e)
            _, distances = Ret.ret_nbrs()
            plot_row(0, distances)

            # benign
            Ret.fit(metric=metric, epoch=e, label=0)
            _, distances = Ret.ret_nbrs()
            plot_row(3, distances, label=0)

            # malignant
            Ret.fit(metric=metric, epoch=e, label=1)
            _, distances = Ret.ret_nbrs()
            plot_row(6, distances, label=1)

#p[-1].legend(run_names)
print('Done distance analysis')

plt.show()