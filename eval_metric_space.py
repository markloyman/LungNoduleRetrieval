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


runs, run_net_types, run_metrics, run_epochs, run_names = load_experiments('DirRating')

# initialize figures

plt.figure("Metric Space - {}".format(dset))
p = [None] * len(metrics) * 5
for i in range(5 * len(metrics)):
    p[i] = plt.subplot(len(metrics), 5, i + 1)

# evaluate

Epochs, Idx_hubness, Idx_symmetry, Idx_concentration, Idx_contrast, Idx_kummar = [], [], [], [], [], []

for m, metric in enumerate(metrics):
    print("Begin: {} metric".format(metric))
    for run, net_type, r, epochs in zip(runs, run_net_types, range(len(runs)), run_epochs):
        plot_data_filename = './Plots/metric-space_{}{}.p'.format(net_type, run)
        try:
            valid_epochs, idx_hubness, idx_symmetry, idx_concentration, idx_contrast, idx_kummar = pickle.load(open(plot_data_filename, 'br'))
            print("Loaded results for {}{}".format(net_type, run))
        except:
            print("Evaluating classification accuracy for {}{}".format(net_type, run))
            # init
            Embed = FileManager.Embed(net_type)
            embed_source = [Embed(run + 'c{}'.format(c), dset) for c in range(n_groups)]
            idx_hubness, idx_symmetry, idx_concentration, idx_contrast, idx_kummar, valid_epochs = [], [], [], [], [], []
            # calculate
            Ret = Retriever(title='{}'.format(run), dset=dset)
            embd, epoch_mask = Ret.load_embedding(embed_source, multi_epcch=True)
            for e in epochs:
                try:
                    Ret.fit(metric=metric, epoch=e)
                    indices, distances = Ret.ret_nbrs()
                    # hubness
                    idx_hubness.append(index.calc_hubness(indices))
                    #   symmetry
                    idx_symmetry.append(index.calc_symmetry(indices))
                    # kumar index
                    tau, l_e = index.kumar(distances, res=0.01)
                    idx_kummar.append(tau)
                    # concentration & contrast
                    idx_concentration.append(index.concentration(distances))
                    idx_contrast.append(index.relative_contrast_imp(distances))
                    valid_epochs.append(e)
                except:
                    print("Epoch {} - no calculated embedding".format(e))
            valid_epochs = np.array(valid_epochs)
            idx_hubness = np.array(list(zip(*idx_hubness)))
            idx_symmetry = np.array(list(zip(*idx_symmetry)))
            idx_concentration = np.array(list(zip(*idx_concentration)))
            idx_contrast = np.array(list(zip(*idx_contrast)))
            idx_kummar = np.array(idx_kummar)
            pickle.dump( (valid_epochs, idx_hubness, idx_symmetry, idx_concentration, idx_contrast, idx_kummar),
                         open(plot_data_filename, 'bw'))
        Epochs += [valid_epochs]
        Idx_hubness += [idx_hubness]
        Idx_symmetry += [idx_symmetry]
        Idx_concentration += [idx_concentration]
        Idx_contrast += [idx_contrast]
        Idx_kummar += [idx_kummar]

# plot

alpha = 0.2


def smooth(signal):
    return savgol_filter(signal, window_length=7, polyorder=1, mode='nearest')

for epochs, idx_hubness, idx_symmetry, idx_concentration, idx_contrast, idx_kummar \
        in zip(Epochs, Idx_hubness, Idx_symmetry, Idx_concentration, Idx_contrast, Idx_kummar):

        #   hubness
        q = p[5 * m + 0].plot(epochs, smooth(idx_hubness[0]))
        p[5 * m + 0].plot(epochs, smooth(idx_hubness[0] + idx_hubness[1]), color=q[0].get_color(), ls='--', alpha=alpha)
        p[5 * m + 0].plot(epochs, smooth(idx_hubness[0] - idx_hubness[1]), color=q[0].get_color(), ls='--', alpha=alpha)
        #   symmetry
        q = p[5 * m + 1].plot(epochs, idx_symmetry[0])
        p[5 * m + 1].plot(epochs, smooth(idx_symmetry[0] + idx_symmetry[1]), color=q[0].get_color(), ls='--', alpha=alpha)
        p[5 * m + 1].plot(epochs, smooth(idx_symmetry[0] - idx_symmetry[1]), color=q[0].get_color(), ls='--', alpha=alpha)
        #   contrast
        q = p[5 * m + 2].plot(epochs, smooth(idx_contrast[0]))
        p[5 * m + 2].plot(epochs, smooth(idx_contrast[0] + idx_contrast[1]), color=q[0].get_color(), ls='--', alpha=alpha)
        p[5 * m + 2].plot(epochs, smooth(idx_contrast[0] - idx_contrast[1]), color=q[0].get_color(), ls='--', alpha=alpha)
        #   concentration
        q = p[5 * m + 3].plot(epochs, smooth(idx_concentration[0]))
        p[5 * m + 3].plot(epochs, smooth(idx_concentration[0] + idx_concentration[1]), color=q[0].get_color(), ls='--', alpha=alpha)
        p[5 * m + 3].plot(epochs, smooth(idx_concentration[0] - idx_concentration[1]), color=q[0].get_color(), ls='--', alpha=alpha)
        #   kumar
        p[5 * m + 4].plot(epochs, smooth(idx_kummar))
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