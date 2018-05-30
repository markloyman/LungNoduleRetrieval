from init import *
from scipy.signal import savgol_filter
from experiments import load_experiments
from Analysis.performance import eval_retrieval, eval_classification


if __name__ == "__main__":

    # Setup

    experiment_name = 'DirRating'
    dset = 'Valid'
    start = timer()

    runs, run_net_types, run_metrics, run_epochs, run_names, _, _ = load_experiments(experiment_name)

    # Initialize Figures

    plt.figure(experiment_name + ' Performance: ' + dset)
    col_titles = ['Classification', 'Retrieval', 'Retrieval']
    col_labels = ['Accuracy', 'Precision', 'Ret-Index']
    legend = []
    for n in run_names:
        legend += [n]
        legend += ['']
        legend += ['']
    rows = 1
    cols = len(col_titles)
    idx  = lambda c, r: c + r*cols
    plt_ = [None]*cols*rows
    for c in range(cols):
        for r in range(rows):
            plt_[idx(c, r)] = plt.subplot(rows, cols, idx(c, r)+1)
            plt_[idx(c, r)].grid(which='major', axis='y')
        plt_[idx(c, 0)].axes.title.set_text(col_titles[c])
        plt_[idx(c, 0)].axes.yaxis.label.set_text(col_labels[c])
    plt_[-2].axes.xaxis.label.set_text("Epochs")

    # Evaluate

    Acc, Acc_std, Prec, Prec_std, Index, Index_std, Valid_epochs = [], [], [], [], [], [], []
    for run, net_type, _, metric, epochs in zip(runs, run_net_types, range(len(runs)), run_metrics, run_epochs):
        plot_data_filename = './Plots/Data/performance_{}{}.p'.format(net_type, run)
        try:
            acc, acc_std, prec, prec_std, index, index_std, valid_epochs = pickle.load(open(plot_data_filename, 'br'))
            print("Loaded results for {}{}".format(net_type, run))
        except:
            print("Evaluating classification accuracy for {}{}".format(net_type, run))
            acc, acc_std, valid_epochs = eval_classification(
                                    run=run, net_type=net_type, dset=dset,
                                    metric=metric, epochs=epochs,
                                    cross_validation=True)
            print("Evaluating retrieval precision for {}{}".format(net_type, run))
            prec, prec_std, index, index_std, _ = eval_retrieval(
                                    run=run, net_type=net_type, dset=dset,
                                    metric=metric, epochs=valid_epochs,
                                    cross_validation=True)

            pickle.dump((acc, acc_std, prec, prec_std, index, index_std, valid_epochs), open(plot_data_filename, 'bw'))

        Acc += [acc]
        Acc_std += [acc_std]
        Prec += [prec]
        Prec_std += [prec_std]
        Index += [index]
        Index_std += [index_std]
        Valid_epochs += [valid_epochs]

    print("Evaluation Done in {:.1f} hours".format((timer() - start) / 60 / 60))

    # Display

    alpha = 0.2

    def smooth(signal):
        return savgol_filter(signal, window_length=7, polyorder=1, mode='nearest')

    for acc, acc_std, prec, prec_std, index, index_std, epochs in zip(Acc, Acc_std, Prec, Prec_std, Index, Index_std, Valid_epochs):
        '''
        # Accuracy
        q = plt_[idx(0, 0)].plot(epochs, acc, '-*')
        plt_[idx(0, 0)].plot(epochs, acc + acc_std, color=q[0].get_color(), ls='--', alpha=alpha)
        plt_[idx(0, 0)].plot(epochs, acc - acc_std, color=q[0].get_color(), ls='--', alpha=alpha)
        Axes.set_ylim(plt_[idx(0, 0)].axes, .8, .9)

        # Precision
        q = plt_[idx(1, 0)].plot(epochs, prec, '-*')
        plt_[idx(1, 0)].plot(epochs, prec + prec_std, color=q[0].get_color(), ls='--', alpha=alpha)
        plt_[idx(1, 0)].plot(epochs, prec - prec_std, color=q[0].get_color(), ls='--', alpha=alpha)
        Axes.set_ylim(plt_[idx(0, 0)].axes, .75, .85)

        # Precision Index
        q = plt_[idx(2, 0)].plot(epochs, index, '-*')
        plt_[idx(2, 0)].plot(epochs, index + index_std, color=q[0].get_color(), ls='--', alpha=alpha)
        plt_[idx(2, 0)].plot(epochs, index - index_std, color=q[0].get_color(), ls='--', alpha=alpha)
        Axes.set_ylim(plt_[idx(0, 0)].axes, .75, .85)
        '''

        # Smoothed
        row = 0

        # Accuracy
        q = plt_[idx(0, row)].plot(epochs, smooth(acc), '-*')
        plt_[idx(0, row)].plot(epochs, smooth(acc + acc_std), color=q[0].get_color(), ls='--', alpha=alpha)
        plt_[idx(0, row)].plot(epochs, smooth(acc - acc_std), color=q[0].get_color(), ls='--', alpha=alpha)
        Axes.set_ylim(plt_[idx(0, row)].axes, .7, .9)

        # Precision
        q = plt_[idx(1, row)].plot(epochs, smooth(prec), '-*')
        plt_[idx(1, row)].plot(epochs, smooth(prec + prec_std), color=q[0].get_color(), ls='--', alpha=alpha)
        plt_[idx(1, row)].plot(epochs, smooth(prec - prec_std), color=q[0].get_color(), ls='--', alpha=alpha)
        Axes.set_ylim(plt_[idx(1, row)].axes, .6, .8)

        # Precision Index
        q = plt_[idx(2, row)].plot(epochs, smooth(index), '-*')
        plt_[idx(2, row)].plot(epochs, smooth(index + index_std), color=q[0].get_color(), ls='--', alpha=alpha)
        plt_[idx(2, row)].plot(epochs, smooth(index - index_std), color=q[0].get_color(), ls='--', alpha=alpha)
        Axes.set_ylim(plt_[idx(2, row)].axes, .6, .8)

    plt_[-1].legend(legend)

    print("Plots Ready in {:.1f} hours".format((timer() - start) / 60 / 60))

    plt.show()

