from init import *
from Analysis import RatingCorrelator, performance
from Network import FileManager
from experiments import load_experiments

# ========================
# Setup
# ========================

exp_name = 'Pooling'
dset = 'Valid'
#X, Y = 'embed', 'rating' #'malig' 'rating'
metrics = ['l2']  #['l2', 'l1', 'cosine']
rating_norm = 'Normal'
n_groups = 5

runs, run_net_types, run_metrics, run_epochs, run_names, _, _ = load_experiments(exp_name)

plt.figure()
plt_ = [None]*len(metrics)*2
for i in range(2 * len(metrics)):
    plt_[i] = plt.subplot(len(metrics), 2, i + 1)

for m, metric_ in enumerate(metrics):

    Valid_epochs, Idx_malig_pearson, Idx_malig_kendall, Idx_rating_pearson, Idx_rating_kendall = [], [], [], [], []

    for run, net_type, dist, epochs, metric in zip(runs, run_net_types, run_metrics, run_epochs, run_metrics):
        plot_data_filename = './Plots/Data/correlation_{}{}.p'.format(net_type, run)
        try:
            valid_epochs, idx_malig_pearson, idx_malig_kendall, idx_rating_pearson, idx_rating_kendall = \
                pickle.load(open(plot_data_filename, 'br'))
            print("Loaded results for {}{}".format(net_type, run))
        except:
            print("Evaluating classification accuracy for {}{} using {}".format(net_type, run, metric))

            Pm, PmStd, Km, KmStd, Pr, PrStd, Kr, KrStd, valid_epochs = \
                performance.eval_correlation(run, net_type, metric, epochs, dset, rating_norm, cross_validation=True)
            idx_malig_pearson = Pm, PmStd
            idx_malig_kendall = Km, KmStd
            idx_rating_pearson= Pr, PrStd
            idx_rating_kendall= Kr, KrStd
            pickle.dump((valid_epochs, idx_malig_pearson, idx_malig_kendall, idx_rating_pearson, idx_rating_kendall),
                        open(plot_data_filename, 'bw'))

            '''
            Embed = FileManager.Embed(net_type)
            embed_source = [Embed(run + 'c{}'.format(c), dset) for c in range(n_groups)]
            idx_malig_pearson, idx_malig_kendall, idx_rating_pearson, idx_rating_kendall \
                = [[] for i in range(n_groups)], [[] for i in range(n_groups)], [[] for i in range(n_groups)], \
                  [[] for i in range(n_groups)]
            valid_epochs = [[] for i in range(n_groups)]
            
            for source in embed_source:
            Reg = RatingCorrelator(W)
            Reg.evaluate_embed_distance_matrix(method=dist)
            Reg.evaluate_rating_space(norm=rating_norm)
            Reg.evaluate_rating_distance_matrix(method=metric)

            p, _, k = Reg.correlate_retrieval(X, Y)
            #p, _, k = Reg.correlate(X, Y)
            P.append(p)
            K.append(k)
        P, S, K = np.array(P), np.array(S), np.array(K)
            '''
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

    for valid_epochs, idx_malig_pearson, idx_malig_kendall, idx_rating_pearson, idx_rating_kendall \
            in zip(Valid_epochs, Idx_malig_pearson, Idx_malig_kendall, Idx_rating_pearson, Idx_rating_kendall):
        #plt.plot(wEpchs, P)
        #plt_[2 * m + 0].plot(K_hist_x, K_hist_y)
        plt_[2 * m + 0].plot(epochs, idx_malig_pearson[0])
        plt_[2 * m + 0].grid(which='major', axis='y')
        plt_[2 * m + 1].plot(epochs, idx_malig_kendall[0])
        plt_[2 * m + 1].grid(which='major', axis='y')
        #labels
        #plt_[2 * m + 0].axes.yaxis.label.set_text(metric)
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