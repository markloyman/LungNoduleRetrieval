from init import *
from Analysis import metric_space_indexes as index
from Analysis.RatingCorrelator import calc_distance_matrix
from Analysis.retrieval import Retriever
from Analysis.performance import mean_cross_validated_index
from Network import FileManager
from experiments import load_experiments
from Analysis.analysis import smooth

# Setup

experiment_name = 'Siam'
dset = 'Valid'
rating_normalizaion = 'Scale' # 'None', 'Normal', 'Scale'
ratiing_metrics = ['euclidean']
n_groups = 5


runs, run_net_types, run_metrics, run_epochs, run_names, _, _ = load_experiments(experiment_name)

# initialize figures

plt.figure("{} Metric Space: {}".format(experiment_name, dset))
p = [None] * len(ratiing_metrics) * 5
for i in range(5 * len(ratiing_metrics)):
    p[i] = plt.subplot(len(ratiing_metrics), 5, i + 1)

# evaluate


Epochs, Idx_hubness, Idx_symmetry, Idx_concentration, Idx_contrast, Idx_kummar = [], [], [], [], [], []

for m, metric_ in enumerate(ratiing_metrics):
    #print("Begin: {} metric".format(metric))
    for run, net_type, r, epochs, metric in zip(runs, run_net_types, range(len(runs)), run_epochs, run_metrics):
        plot_data_filename = './Plots/Data/metric-space_{}{}.p'.format(net_type, run)
        try:
            combined_epochs, idx_hubness, idx_symmetry, idx_concentration, idx_contrast, idx_kummar = pickle.load(open(plot_data_filename, 'br'))
            idx_kummar = np.reshape(idx_kummar, (1, np.max(idx_kummar.shape)))
            print("Loaded results for {}{}".format(net_type, run))
        except:
            print("Evaluating classification accuracy for {}{} using {}".format(net_type, run, metric))
            # init
            Embed = FileManager.Embed(net_type)
            embed_source = [Embed(run + 'c{}'.format(c), dset) for c in range(n_groups)]
            idx_hubness, idx_symmetry, idx_concentration, idx_contrast, idx_kummar \
                = [[] for i in range(n_groups)], [[] for i in range(n_groups)], [[] for i in range(n_groups)], [[] for i in range(n_groups)], [[] for i in range(n_groups)]
            valid_epochs = [[] for i in range(n_groups)]
            # calculate
            Ret = Retriever(title='{}'.format(run), dset=dset)
            for i, source in enumerate(embed_source):
                embd, epoch_mask = Ret.load_embedding(source, multi_epcch=True)

                for e in epochs:
                    try:
                        Ret.fit(metric=metric, epoch=e)
                        indices, distances = Ret.ret_nbrs()
                        # hubness
                        idx_hubness[i].append(index.calc_hubness(indices))
                        #   symmetry
                        idx_symmetry[i].append(index.calc_symmetry(indices))
                        # kumar index
                        tau, l_e = index.kumar(distances, res=0.01)
                        idx_kummar[i].append(tau)
                        # concentration & contrast
                        idx_concentration[i].append(index.concentration(distances))
                        idx_contrast[i].append(index.relative_contrast_imp(distances))
                        valid_epochs[i].append(e)
                    except:
                        print("Epoch {} - no calculated embedding".format(e))
                valid_epochs[i] = np.array(valid_epochs[i])
                idx_hubness[i] = np.array(list(zip(*idx_hubness[i])))
                idx_symmetry[i] = np.array(list(zip(*idx_symmetry[i])))
                idx_concentration[i] = np.array(list(zip(*idx_concentration[i])))
                idx_contrast[i] = np.array(list(zip(*idx_contrast[i])))
                idx_kummar[i] = np.array([idx_kummar[i]])

            combined_epochs = [i for i, c in enumerate(np.bincount(np.concatenate(valid_epochs))) if c > 3]

            idx_hubness = mean_cross_validated_index(idx_hubness, valid_epochs, combined_epochs)
            idx_symmetry = mean_cross_validated_index(idx_symmetry, valid_epochs, combined_epochs)
            idx_concentration = mean_cross_validated_index(idx_concentration, valid_epochs, combined_epochs)
            idx_contrast = mean_cross_validated_index(idx_contrast, valid_epochs, combined_epochs)
            idx_kummar = mean_cross_validated_index(idx_kummar, valid_epochs, combined_epochs)

            #idx_hubness = np.mean(idx_hubness, axis=0)
            #idx_symmetry = np.mean(idx_symmetry, axis=0)
            #idx_concentration = np.mean(idx_concentration, axis=0)
            #idx_contrast = np.mean(idx_contrast, axis=0)
            #idx_kummar = np.mean(idx_kummar, axis=0)
            pickle.dump( (combined_epochs, idx_hubness, idx_symmetry, idx_concentration, idx_contrast, idx_kummar),
                         open(plot_data_filename, 'bw'))
        Epochs += [combined_epochs]
        Idx_hubness += [idx_hubness]
        Idx_symmetry += [idx_symmetry]
        Idx_concentration += [idx_concentration]
        Idx_contrast += [idx_contrast]
        Idx_kummar += [idx_kummar]

# plot

alpha = 0.2

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
        p[5 * m + 4].plot(epochs, smooth(idx_kummar[0]))
        # labels
        if r == 0:  # first column
            p[5 * m + 0].axes.yaxis.label.set_text(metric)
        if m == 0:  # first row
            p[5 * m + 0].axes.title.set_text('hubness')
            p[5 * m + 1].axes.title.set_text('symmetry')
            p[5 * m + 2].axes.title.set_text('contrast')
            p[5 * m + 3].axes.title.set_text('concentration')
            p[5 * m + 4].axes.title.set_text('kumari')
        if m == len(ratiing_metrics) - 1:  # last row
            p[5 * m + 2].axes.xaxis.label.set_text('epochs')
p[-1].legend(run_names)
print('Done doMetricSpaceIndexes')

plt.show()