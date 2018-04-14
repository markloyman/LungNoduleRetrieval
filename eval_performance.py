from init import *
from functools import reduce
from scipy.signal import savgol_filter
from Analysis import Retriever
from Network import FileManager
from experiments import load_experiments


def accuracy(true, pred):
    pred = np.clip(pred, 0, 1)
    pred = np.squeeze(np.round(pred).astype('uint'))
    mask = (true==pred).astype('uint')
    acc = np.mean(mask)
    return acc


def precision(query, nn):
    mask = (query == nn).astype('uint')
    np.sum(mask)
    assert(False)


def eval_classification(run, net_type, metric, epochs, dset, NN=[3, 5, 7, 11, 17], cross_validation=False, n_groups=5):
    Embed = FileManager.Embed(net_type)
    Pred_L1O = []

    if cross_validation:
        # Load
        embed_source= [Embed(run + 'c{}'.format(c), dset) for c in range(n_groups)]
        Ret = Retriever(title='{}-{}'.format(net_type, run), dset=dset)
        Ret.load_embedding(embed_source, multi_epcch=True)
        valid_epochs = []
        for E in epochs:
            # Calc
            pred_l1o = []
            try:
                for N in NN:
                    pred_l1o.append(Ret.classify_leave1out(epoch=E, n=N)[1])
                Pred_L1O.append(np.array(pred_l1o))
                valid_epochs.append(E)
            except:
                print("Epoch {} - no calculated embedding".format(E))
        Pred_L1O = np.array(Pred_L1O)
    else:
        for E in epochs:
            # Load
            embed_source = Embed(run, E, dset)
            Ret = Retriever(title='{}-{}'.format(net_type, run), dset=dset)
            Ret.load_embedding(embed_source)
            # Calc
            pred_l1o = []
            for N in NN:
                pred_l1o.append(Ret.classify_leave1out(n=N)[1])
            Pred_L1O.append(np.array(pred_l1o))
        Pred_L1O = np.array(Pred_L1O)

    return np.mean(Pred_L1O, axis=-1), np.std(Pred_L1O, axis=-1), valid_epochs


def eval_retrieval(run, net_type, metric, epochs, dset, NN=[3, 5, 7, 11, 17], cross_validation=False, n_groups=5):
    Embed = FileManager.Embed(net_type)
    Prec, Prec_b, Prec_m = [], [], []

    if cross_validation:
        # Load
        embed_source= [Embed(run + 'c{}'.format(c), dset) for c in range(n_groups)]
        Ret = Retriever(title='{}-{}'.format(net_type, run), dset=dset)
        Ret.load_embedding(embed_source, multi_epcch=True)
        valid_epochs = []
        for E in epochs:
            # Calc
            prec, prec_b, prec_m = [], [], []
            try:
                Ret.fit(np.max(NN), metric=metric, epoch=E)
                for N in NN:
                    p, pb, pm = Ret.evaluate_precision(n=N)
                    prec.append(p)
                    prec_b.append(pb)
                    prec_m.append(pm)
                Prec.append(np.array(prec))
                Prec_b.append(np.array(prec_b))
                Prec_m.append(np.array(prec_m))
                valid_epochs.append(E)
            except:
                print("Epoch {} - no calculated embedding".format(E))
    else:
        for E in epochs:
            Ret = Retriever(title='', dset='')
            if cross_validation:
                embed_source = [Embed(run + 'c{}'.format(c), E, dset) for c in range(n_groups)]
            else:
                embed_source = Embed(run, E, dset)
            Ret.load_embedding(embed_source)

            prec, prec_b, prec_m = [], [], []
            Ret.fit(np.max(NN), metric=metric)
            for N in NN:
                p, pm, pb = Ret.evaluate_precision(n=N)
                prec.append(p)
                prec_b.append(pb)
                prec_m.append(pm)
            Prec.append(np.array(prec))
            Prec_b.append(np.array(prec_b))
            Prec_m.append(np.array(prec_m))

    # Pred_L1O = np.transpose(np.array(Pred_L1O))
    Prec = (np.array(Prec))
    Prec_m = (np.array(Prec_m))
    Prec_b = (np.array(Prec_b))
    f1 = 2 * Prec_b * Prec_m / (Prec_b + Prec_m)

    return np.mean(Prec, axis=-1), np.std(Prec, axis=-1), np.mean(f1, axis=-1), np.std(f1, axis=-1), valid_epochs


if __name__ == "__main__":

    # Setup

    dset = 'Valid'
    start = timer()

    runs, run_net_types, run_metrics, run_epochs, run_names = load_experiments('NewNetwork')

    # Initialize Figures

    plt.figure('Performance - ' + dset)
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
        plot_data_filename = './Plots/performance_{}{}.p'.format(net_type, run)
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

            pickle.dump((Acc, Acc_std, Prec, Prec_std, Index, Index_std, Valid_epochs), open(plot_data_filename, 'bw'))

        Acc += [acc]
        Acc_std += [acc_std]
        Prec += [prec]
        Prec_std += [prec_std]
        Index += [index]
        Index_std += [index_std]
        Valid_epochs += [valid_epochs]

    print("Evaluation Done in {:.1f} hours".format((timer() - start) / 60 / 60))

    # Display

    alpha = 0.6

    def smooth(signal):
        return savgol_filter(signal, window_length=5, polyorder=2, mode='nearest')

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

