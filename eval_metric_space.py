#from eval_metric_space import idx
from init import *
from Analysis import metric_space_indexes as index
from Analysis.RatingCorrelator import calc_distance_matrix
from Analysis.retrieval import Retriever
from Analysis.performance import mean_cross_validated_index
from Network import FileManager
from Analysis.metric_space_indexes import eval_embed_space
from experiments import load_experiments, CrossValidationManager
from Analysis.analysis import smooth


# Setup

experiment_name = 'DirRD-Semi-Supervised'
dset = 'Test'
rating_normalizaion = 'Scale' # 'None', 'Normal', 'Scale'
ratiing_metrics = ['euclidean']
n_groups = 5


runs, run_net_types, run_metrics, run_epochs, run_names, _, _ = load_experiments(experiment_name)

alpha = 0.3

USE_CACHE = False
DUMP_CACHE = False

cv = CrossValidationManager('RET')
configurations = ['{}{}'.format(cv.get_run_id(i)[0], cv.get_run_id(i)[1]) for i in ([0, 1, 3, 4, 7] if dset == 'Valid' else [2, 5, 6, 8, 9])]  # [range(10)]
# configurations = range(n_groups)
# configurations = [1]
dset = 'Valid'

#indexes = ['Hubness', 'Symmetry', 'Contrast', 'Concentration', 'Kumari']
indexes = ['Hubness', 'Symmetry']  # 'FeatCorr', 'SampCorr'
M = len(indexes)

# initialize figures

plt.figure("{} Metric Space: {}".format(experiment_name, dset))
p = [None] * len(ratiing_metrics) * M
for i in range(M * len(ratiing_metrics)):
    p[i] = plt.subplot(len(ratiing_metrics), M, i + 1)

legend = []
for n in run_names:
    legend += [n]
    legend += ['']
    legend += ['']

# evaluate

Epochs, Idx_hubness, Idx_symmetry, Idx_concentration, Idx_contrast, Idx_kummar, Idx_featCorr, Idx_sampCorr = [], [], [], [], [], [], [], []

for m, metric_ in enumerate(ratiing_metrics):
    #print("Begin: {} metric".format(metric))
    for run, net_type, r, epochs, metric in zip(runs, run_net_types, range(len(runs)), run_epochs, run_metrics):
        plot_data_filename = './Plots/Data/metric-space_{}{}.p'.format(net_type, run)
        try:
            if USE_CACHE is False:
                print('WARNINING - SKIIPING TO CALCULATION')
                assert False

            combined_epochs, idx_hubness, idx_symmetry, idx_concentration, idx_contrast, idx_kummar, idx_featCorr, idx_sampCorr \
                = pickle.load(open(plot_data_filename, 'br'))

            #idx_kummar = np.reshape(idx_kummar, (1, np.max(idx_kummar.shape)))
            print("Loaded results for {}{}".format(net_type, run))

        except:
            print("Evaluating classification accuracy for {}{} using {}".format(net_type, run, metric))

            Embed = FileManager.Embed(net_type)
            embed_source = [Embed(run + 'c{}'.format(c), dset) for c in configurations]

            combined_epochs, idx_hubness, idx_symmetry, idx_concentration, idx_contrast, idx_kummar, idx_featCorr, idx_sampCorr = \
                eval_embed_space(embed_source, metric, 'euclidean', epochs, dset)

            if DUMP_CACHE:
                pickle.dump((combined_epochs, idx_hubness, idx_symmetry, idx_concentration, idx_contrast, idx_kummar,
                             idx_featCorr, idx_sampCorr),
                            open(plot_data_filename, 'bw'))
            else:
                print('NO DUMP')

        Epochs += [combined_epochs]
        Idx_hubness += [idx_hubness]
        Idx_symmetry += [idx_symmetry]
        Idx_concentration += [idx_concentration]
        Idx_contrast += [idx_contrast]
        Idx_kummar += [idx_kummar]
        Idx_featCorr += [idx_featCorr]
        Idx_sampCorr += [idx_sampCorr]

# plot

for epochs, idx_hubness, idx_symmetry, idx_concentration, idx_contrast, idx_kummar, idx_featCorr, idx_sampCorr \
        in zip(Epochs, Idx_hubness, Idx_symmetry, Idx_concentration, Idx_contrast, Idx_kummar, Idx_featCorr, Idx_sampCorr):

        #   hubness
        next_plot = 0
        if 'Hubness' in indexes:
            q = p[M * m + next_plot].plot(epochs, smooth(idx_hubness[0]), marker='x')  #
            p[M * m + next_plot].plot(epochs, smooth(idx_hubness[0] + idx_hubness[1]), color=q[0].get_color(), ls='--', alpha=alpha)
            p[M * m + next_plot].plot(epochs, smooth(idx_hubness[0] - idx_hubness[1]), color=q[0].get_color(), ls='--', alpha=alpha)
            next_plot += 1

        #   symmetry
        if 'Symmetry' in indexes:
            q = p[M * m + next_plot].plot(epochs, smooth(idx_symmetry[0]))
            p[M * m + next_plot].plot(epochs, smooth(idx_symmetry[0] + idx_symmetry[1]), color=q[0].get_color(), ls='--', alpha=alpha)
            p[M * m + next_plot].plot(epochs, smooth(idx_symmetry[0] - idx_symmetry[1]), color=q[0].get_color(), ls='--', alpha=alpha)
            next_plot += 1

        #   contrast
        if 'Contrast' in indexes:
            q = p[M * m + next_plot].plot(epochs, smooth(idx_contrast[0]))
            p[M * m + next_plot].plot(epochs, smooth(idx_contrast[0] + idx_contrast[1]), color=q[0].get_color(), ls='--', alpha=alpha)
            p[M * m + next_plot].plot(epochs, smooth(idx_contrast[0] - idx_contrast[1]), color=q[0].get_color(), ls='--', alpha=alpha)
            next_plot += 1

        #   concentration
        if 'Concentration' in indexes:
            q = p[M * m + next_plot].plot(epochs, smooth(idx_concentration[0]))
            p[M * m + next_plot].plot(epochs, smooth(idx_concentration[0] + idx_concentration[1]), color=q[0].get_color(), ls='--', alpha=alpha)
            p[M * m + next_plot].plot(epochs, smooth(idx_concentration[0] - idx_concentration[1]), color=q[0].get_color(), ls='--', alpha=alpha)
            next_plot += 1

        #   kumar
        if 'Kumari' in indexes:
            p[M * m + next_plot].plot(epochs, smooth(idx_kummar[0]))
            next_plot += 1

        if 'FeatCorr' in indexes:
            p[M * m + next_plot].plot(epochs, smooth(idx_featCorr[0]))
            next_plot += 1

        if 'SampCorr' in indexes:
            p[M * m + next_plot].plot(epochs, smooth(idx_sampCorr[0]))
            next_plot += 1

        # labels
        #if r == 0:  # first column
        #    p[M * m + 0].axes.yaxis.label.set_text(metric)
        if m == 0:  # first row
            for i, label in enumerate(indexes):
                p[M * m + i].axes.title.set_text(label)
            #p[M * m + 0].axes.title.set_text('Hubness')
            #p[M * m + 1].axes.title.set_text('Symmetry')
            #p[M * m + 2].axes.title.set_text('Contrast')
            #p[M * m + 3].axes.title.set_text('Concentration')
            #p[M * m + 4].axes.title.set_text('Kumari')
        if m == len(ratiing_metrics) - 1:  # last row
            p[M * m + M//2].axes.xaxis.label.set_text('Epochs')
p[1].legend(legend if indexes[-1] is not 'Kumari' else run_names)

for i in range(M * len(ratiing_metrics)):
    p[i].grid(which='both', axis='y')

print('Done doMetricSpaceIndexes')

plt.show()