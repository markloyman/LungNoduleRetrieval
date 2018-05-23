import numpy as np
from Analysis import Retriever, RatingCorrelator
from Network import FileManager


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
                    pred_l1o.append(Ret.classify_kfold(epoch=E, n=N, k_fold=10, metric=metric))
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
                pred_l1o.append(Ret.classify_leave1out(n=N, metric=metric)[1])
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


def eval_correlation(run, net_type, metric, epochs, dset, rating_norm='none', cross_validation=False, n_groups=5):

    Embed = FileManager.Embed(net_type)

    if cross_validation:
        # Load
        embed_source= [Embed(run + 'c{}'.format(c), dset) for c in range(n_groups)]

        Reg = RatingCorrelator(embed_source, multi_epoch=True)

        valid_epochs = []
        Pm, Km, Pr, Kr = [], [], [], []
        PmStd, KmStd, PrStd, KrStd = [], [], [], []
        for E in epochs:
            # Calc
            try:
                Reg.evaluate_embed_distance_matrix(method=metric, epoch=E)
                Reg.evaluate_rating_space(norm=rating_norm)
                Reg.evaluate_rating_distance_matrix(method=metric, clustered_rating_distance=False)

                pm, _, km = Reg.correlate_retrieval('embed', 'malig')
                pr, _, kr = Reg.correlate_retrieval('embed', 'rating')

                valid_epochs.append(E)
            except:
                print("Epoch {} - no calculated embedding".format(E))
                continue
            Pm.append(pm[0])
            Km.append(km[0])
            Pr.append(pr[0])
            Kr.append(kr[0])
            PmStd.append(pm[1])
            KmStd.append(km[1])
            PrStd.append(pr[1])
            KrStd.append(kr[1])
    else:
        assert False
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

    Pm, Km, Pr, Kr = np.array(Pm), np.array(Km), np.array(Pr), np.array(Kr)
    PmStd, KmStd, PrStd, KrStd = np.array(PmStd), np.array(KmStd), np.array(PrStd), np.array(KrStd)

    return Pm, PmStd, Km, KmStd, Pr, PrStd, Kr, KrStd, valid_epochs