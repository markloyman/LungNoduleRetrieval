from tables.idxutils import col_light

from init import *
from Analysis import Retriever
from matplotlib.axes import Axes
from Network.data_loader import load_nodule_raw_dataset
from Analysis.RatingCorrelator import calc_distance_matrix
import Analysis.metric_space_indexes as index
import FileManager


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


def eval_classification(run, net_type, metric, epochs, dset):
    NN = [3, 5, 7, 11, 17]
    Embed = FileManager.Embed(net_type)
    WW = [Embed(run, E, dset) for E in epochs]

    Pred_L1O = []
    for W in WW:
        # Load
        Ret = Retriever(title='{}-{}'.format(net_type, run), dset=dset)
        Ret.load_embedding(W)
        # Calc
        pred_l1o = []
        for N in NN:
            Ret.fit(N, metric=metric)
            pred_l1o.append(Ret.classify_leave1out()[1])
        Pred_L1O.append(np.array(pred_l1o))
    Pred_L1O = np.array(Pred_L1O)

    return np.mean(Pred_L1O, axis=-1), np.std(Pred_L1O, axis=-1)


def eval_retrieval(run, net_type, metric, epochs, dset):
    NN = [3, 5, 7, 11, 17]
    Embed = FileManager.Embed(net_type)
    Prec, Prec_b, Prec_m = [], [], []
    WW = [Embed(run, E, dset) for E in epochs]

    for W in WW:
        Ret = Retriever(title='', dset='')
        Ret.load_embedding(W)

        prec, prec_b, prec_m = [], [], []
        for N in NN:
            Ret.fit(N, metric=metric)
            p = Ret.evaluate_precision(plot=False, split=False)
            pm, pb = Ret.evaluate_precision(plot=False, split=True)
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

    return np.mean(Prec, axis=-1), np.std(Prec, axis=-1), np.mean(f1, axis=-1), np.std(f1, axis=-1)


if __name__ == "__main__":

# Setup

    dset = 'Valid'

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
    #runs = ['100', '016XXXX', '021', '023X']  #['064X', '078X', '026'] #['064X', '071' (is actually 071X), '078X', '081', '082']
    #run_net_types = ['siam', 'trip','trip', 'trip']  #, 'dir']
    runs            = ['021', '022XX', '023X', '025']
    run_names       = ['max-pool', 'rmac', 'categ', 'confidence+cat' ]
    run_net_types   = ['trip']*len(runs)
    run_metrics     = ['l2']*len(runs)
    #rating_normalizaion = 'Scale' # 'None', 'Normal', 'Scale'

    run_epochs = [ [5, 15, 25, 35],
                   [5, 15, 25, 35, 45, 55],
                   [5, 15, 25, 35],
                   [5, 15, 25, 35, 45, 55]
               ]
    '''

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

    for run, net_type, _, metric, epochs in zip(runs, run_net_types, range(len(runs)), run_metrics, run_epochs):
        acc, acc_std = eval_classification(run=run, net_type=net_type, metric=metric, epochs=epochs, dset=dset)
        q = plt_[idx(0, 0)].plot(epochs, acc, '-*')
        plt_[idx(0, 0)].plot(epochs, acc + acc_std, color=q[0].get_color(), ls='--')
        plt_[idx(0, 0)].plot(epochs, acc - acc_std, color=q[0].get_color(), ls='--')
        Axes.set_ylim(plt_[idx(0, 0)].axes, .8, .9)

        prec, prec_std, index, index_std = eval_retrieval(run=run, net_type=net_type, metric=metric, epochs=epochs, dset=dset)
        q = plt_[idx(1, 0)].plot(epochs, prec, '-*')
        plt_[idx(1, 0)].plot(epochs, prec + prec_std, color=q[0].get_color(), ls='--')
        plt_[idx(1, 0)].plot(epochs, prec - prec_std, color=q[0].get_color(), ls='--')
        Axes.set_ylim(plt_[idx(0, 0)].axes, .75, .85)

        q = plt_[idx(2, 0)].plot(epochs, index, '-*')
        plt_[idx(2, 0)].plot(epochs, index + index_std, color=q[0].get_color(), ls='--')
        plt_[idx(2, 0)].plot(epochs, index - index_std, color=q[0].get_color(), ls='--')
        Axes.set_ylim(plt_[idx(0, 0)].axes, .75, .85)

    plt_[-1].legend(legend)

    print('Plots Ready')
    plt.show()

