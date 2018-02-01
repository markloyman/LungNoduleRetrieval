from init import *
from Analysis import Retriever
from Network.data import load_nodule_raw_dataset
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


def eval_classification(runs, net_types, run_metrics, epochs, dset):
    plt.figure('KNN Classification - ' + dset)
    for run, net_type, idx, metric in zip(runs, net_types, range(len(runs)), run_metrics):
        NN = [3, 5, 7, 11, 17]
        Embed = FileManager.Embed(net_type)
        WW = [Embed(run, E, dset) for E in wEpchs]

        Pred_L1O = []
        for W in WW:
            Ret = Retriever(title='{}-{}'.format(net_type, run), dset=dset)
            Ret.load_embedding(W)

            pred_l1o = []
            for N in NN:
                Ret.fit(N, metric=metric)
                pred_l1o.append(Ret.classify_leave1out()[1])

            Pred_L1O.append(np.array(pred_l1o))

        Pred_L1O = (np.array(Pred_L1O))
        plt.subplot(1, len(wRuns), idx + 1)
        plt.plot(wEpchs, Pred_L1O, '-*')
        plt.grid(which='major', axis='y')
        plt.title('{}-{}'.format(net_type, run))
        plt.ylabel('ACC')
        plt.xlabel('epoch')
        plt.legend(NN)
    print('Done Classification.')


def eval_retrieval(runs, net_types, run_metrics, epochs, dset):
    NN = [3, 5, 7, 11, 17]
    for run, net_type, idx, metric in zip(runs, net_types, range(len(runs)), run_metrics):
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

        plt.figure('RET_' + run + '_' + dset)

        plt.subplot(211)
        plt.plot(wEpchs, Prec, '-*')
        plt.legend(NN)
        plt.title('Retrieval')
        plt.grid(which='major', axis='y')
        plt.ylim([0.7, 0.9])
        plt.ylabel('Precision')
        plt.xlabel('epoch')

        f1 = 2 * Prec_b * Prec_m / (Prec_b + Prec_m)
        plt.subplot(212)
        plt.plot(wEpchs, f1, '-*')
        plt.legend(NN)
        plt.title('F1')
        plt.grid(which='major', axis='y')
        plt.ylim([0.7, 0.9])
        plt.ylabel('Retrieval Index')
        plt.xlabel('epoch')

        # plt.subplot(325)
        # plt.grid(which='major', axis='y')
        # plt.plot(wEpchs, Prec_b, '-*')
        # plt.legend(NN)
        # plt.title('Benign')
        # plt.ylim([0.6, 1.0])
        # plt.ylabel('Precision')
        # plt.xlabel('epoch')

        # plt.subplot(326)
        # plt.plot(wEpchs, Prec_m, '-*')
        # plt.legend(NN)
        # plt.title('Malignant')
        # plt.ylim([0.6, 1.0])
        # plt.grid(which='major', axis='y')
        # plt.ylabel('Precision')
        # plt.xlabel('epoch')

    print('Done Retrieval.')


if __name__ == "__main__":

    dset = 'Valid'
    wRuns = ['011XXX']  #['064X', '078X', '026'] #['064X', '071' (is actually 071X), '078X', '081', '082']
    wRunsNet = ['trip']  #, 'dir']
    wRunMetrics = ['l2']

    #wRuns = ['103']
    #wRunsNet = ['dir']  # , 'dir']
    #run_metrics = ['l2']

    #rating_normalizaion = 'Scale' # 'None', 'Normal', 'Scale'

    wEpchs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

    #leg = ['E{}'.format(E) for E in wEpchs]

    eval_classification(runs=wRuns, net_types=wRunsNet, run_metrics=wRunMetrics, epochs=wEpchs, dset=dset)
    eval_retrieval(runs=wRuns, net_types=wRunsNet, run_metrics=wRunMetrics, epochs=wEpchs, dset=dset)

    plt.show()



