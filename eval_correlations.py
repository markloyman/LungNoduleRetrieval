from init import *
from experiments import load_experiments, CrossValidationManager
from Analysis import RatingCorrelator, performance
from Analysis.analysis import smooth
from Network import FileManager

# ========================
# Setup
# ========================

experiment_name = 'DirRD-Semi-Supervised'
dset = 'Test'
rating_metrics = ['euclidean']  #['l2', 'l1', 'cosine']
rating_norm = 'none'
n_groups = 5
objective = 'rating'  # 'size'
alpha = 0.3

USE_CACHE = False
DUMP_CACHE = False

cv = CrossValidationManager('RET')
configurations = ['{}{}'.format(cv.get_run_id(i)[0], cv.get_run_id(i)[1]) for i in ([0, 1, 3, 4, 7] if dset == 'Valid' else [2, 5, 6, 8, 9])]  # [range(10)]
#configurations = range(n_groups)
# configurations = [1]
dset = 'Valid'

runs, run_net_types, run_metrics, run_epochs, run_names, _, _ = load_experiments(experiment_name)

for m, metric_rating in enumerate(rating_metrics):

    Valid_epochs, Idx_malig_pearson, Idx_malig_kendall, Idx_rating_pearson, Idx_rating_kendall = [], [], [], [], []

    for run, net_type, dist, epochs, metric in zip(runs, run_net_types, run_metrics, run_epochs, run_metrics):
        plot_data_filename = './Plots/Data/correlation_{}{}.p'.format(net_type, run)
        try:
            if USE_CACHE is False:
                print('NOTE: SKIPPING TO EVELUATION')
                assert False
            valid_epochs, idx_malig_pearson, idx_malig_kendall, idx_rating_pearson, idx_rating_kendall = \
                pickle.load(open(plot_data_filename, 'br'))
            print("Loaded results for {}{}".format(net_type, run))

        except:
            print("Evaluating classification accuracy for {}{} using {}".format(net_type, run, metric))

            # Load
            Embed = FileManager.Embed(net_type)
            embed_source = [Embed(run + 'c{}'.format(c), dset) for c in configurations]

            Pm, PmStd, Km, KmStd, Pr, PrStd, Kr, KrStd, valid_epochs = \
                performance.eval_correlation(embed_source, metric, metric_rating, epochs,
                                             objective=objective, rating_norm=rating_norm,
                                             seq=False, local_scaling=False)
            idx_malig_pearson = Pm, PmStd
            idx_malig_kendall = Km, KmStd
            idx_rating_pearson= Pr, PrStd
            idx_rating_kendall= Kr, KrStd

            if DUMP_CACHE:
                pickle.dump(
                    (valid_epochs, idx_malig_pearson, idx_malig_kendall, idx_rating_pearson, idx_rating_kendall),
                    open(plot_data_filename, 'bw'))
            else:
                print('NO DUMP!!')

        Idx_malig_pearson += [idx_malig_pearson]
        Idx_malig_kendall += [idx_malig_kendall]
        Idx_rating_pearson += [idx_rating_pearson]
        Idx_rating_kendall += [idx_rating_kendall]
        Valid_epochs += [valid_epochs]

        # correlation distribution
        #for W in WW[-1:]:
        #    Reg = RatingCorrelator(W)
        #    Reg.evaluate_embed_distance_matrix(method=dist)
        #    Reg.evaluate_rating_space(norm=rating_norm)
        #    Reg.evaluate_rating_distance_matrix(method=metric)
        #    K_hist_x, K_hist_y = Reg.kendall_histogram(X, Y)

    #   Plot
    # ==============

    # setup
    do_kendall = False
    n_rows = 2 if do_kendall else 1
    n_cols = 2
    n_cells = n_rows*n_cols
    # initialize
    plt.figure(experiment_name + ' Correlation: ' + dset)
    plt_ = [None] * len(rating_metrics) * n_cells
    for i in range(n_cells):
        plt_[i] = plt.subplot(n_rows, n_cols, i + 1)
        #plt_[i].grid(which='major', axis='y')
    legend = []
    for n in run_names:
        legend += [n]
        legend += ['']
        legend += ['']

    for valid_epochs, idx_malig_pearson, idx_malig_kendall, idx_rating_pearson, idx_rating_kendall \
            in zip(Valid_epochs, Idx_malig_pearson, Idx_malig_kendall, Idx_rating_pearson, Idx_rating_kendall):

        # Malignancy Pearson Correlation
        q = plt_[2 * m + 0].plot(valid_epochs, smooth(idx_malig_pearson[0]))
        plt_[2 * m + 0].plot(valid_epochs, smooth(idx_malig_pearson[0] + idx_malig_pearson[1]),
                             color=q[0].get_color(), ls='--', alpha=alpha)
        plt_[2 * m + 0].plot(valid_epochs, smooth(idx_malig_pearson[0] - idx_malig_pearson[1]),
                             color=q[0].get_color(), ls='--', alpha=alpha)
        plt_[2 * m + 0].grid(which='major', axis='y')

        # Malignancy Kendall Correlation
        if do_kendall:
            q = plt_[2 * m + 2].plot(valid_epochs, smooth(idx_malig_kendall[0]))
            plt_[2 * m + 2].plot(valid_epochs, smooth(idx_malig_kendall[0] + idx_malig_kendall[1]),
                                 color=q[0].get_color(), ls='--', alpha=alpha)
            plt_[2 * m + 2].plot(valid_epochs, smooth(idx_malig_kendall[0] - idx_malig_kendall[1]),
                                 color=q[0].get_color(), ls='--', alpha=alpha)
            plt_[2 * m + 2].grid(which='major', axis='y')

        # Rating Pearson Correlation
        q = plt_[2 * m + 1].plot(valid_epochs, smooth(idx_rating_pearson[0]), marker='.')
        plt_[2 * m + 1].plot(valid_epochs, smooth(idx_rating_pearson[0] + idx_rating_pearson[1]),
                             color=q[0].get_color(), ls='--', alpha=alpha)
        plt_[2 * m + 1].plot(valid_epochs, smooth(idx_rating_pearson[0] - idx_rating_pearson[1]),
                             color=q[0].get_color(), ls='--', alpha=alpha)
        plt_[2 * m + 1].grid(which='major', axis='y')

        # Rating Kendall Correlation
        if do_kendall:
            q = plt_[2 * m + 3].plot(valid_epochs, smooth(idx_rating_kendall[0]))
            plt_[2 * m + 3].plot(valid_epochs, smooth(idx_rating_kendall[0] + idx_rating_kendall[1]),
                                 color=q[0].get_color(), ls='--', alpha=alpha)
            plt_[2 * m + 3].plot(valid_epochs, smooth(idx_rating_kendall[0] - idx_rating_kendall[1]),
                                 color=q[0].get_color(), ls='--', alpha=alpha)
            plt_[2 * m + 3].grid(which='major', axis='y')

        if m == 0:  # first row
            plt_[0].axes.title.set_text('Malignancy' if objective == 'rating' else 'Size')
            plt_[1].axes.title.set_text('Ratings')
            plt_[0].axes.yaxis.label.set_text('Pearson')
            if do_kendall:
                plt_[2].axes.yaxis.label.set_text('Kendall')
        if m == len(rating_metrics) - 1:  # last row
            plt_[n_rows*m+1].axes.xaxis.label.set_text('epochs')
            plt_[n_rows*m+1].legend(legend)

    for i in range(n_cells):
        #plt_[i] = plt.subplot(n_rows, n_cols, i + 1)
        plt_[i].grid(which='both', axis='y')
        plt_[i].axes.set_ylim(.0, .5)

print('Plots Ready...')
plt.show()