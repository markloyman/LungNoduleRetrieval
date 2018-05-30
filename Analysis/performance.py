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


def merge_epochs(valid_epochs, min_element=4):
    combined_epochs = [i for i, c in enumerate(np.bincount(np.concatenate(valid_epochs))) if c >= min_element]
    return combined_epochs


def mean_cross_validated_index(index, valid_epochs, combined_epochs):
    merged = np.zeros((len(combined_epochs)))
    merged_std = np.zeros((len(combined_epochs)))
    for ep_id, epoch in enumerate(combined_epochs):
        #collect = [[np.array(idx)[:, i] for i, e in enumerate(ve) if e == epoch] for idx, ve in zip(index, valid_epochs) if epoch in ve]
        #collect = [np.array(idx)[np.argwhere(ve==epoch)[0]] for idx, ve in zip(index, valid_epochs) if epoch in ve]
        collect = []
        for idx, ve in zip(index, valid_epochs):
            if epoch in ve:
                for i, e in enumerate(ve):
                    if e == epoch:
                        collect += [idx[i, :]]
        merged[ep_id] = np.mean(np.mean(collect, axis=-1), axis=0)
        merged_std[ep_id] = np.mean(np.std(collect, axis=-1), axis=0)
    return merged, merged_std


def eval_classification(run, net_type, metric, epochs, dset, NN=[7, 11, 17], cross_validation=False, n_groups=5):
    Embed = FileManager.Embed(net_type)
    Pred_L1O = [[] for i in range(n_groups)]
    valid_epochs = [[] for i in range(n_groups)]
    if cross_validation:
        # Load
        embed_source= [Embed(run + 'c{}'.format(c), dset) for c in range(n_groups)]
        Ret = Retriever(title='{}-{}'.format(net_type, run), dset=dset)
        for i, source in enumerate(embed_source):
            Ret.load_embedding([source], multi_epcch=True)
            for E in epochs:
                # Calc
                pred_l1o = []
                try:
                    for N in NN:
                        pred_l1o.append(Ret.classify_kfold(epoch=E, n=N, k_fold=10, metric=metric))
                    Pred_L1O[i].append(np.array(pred_l1o))
                    valid_epochs[i].append(E)
                except:
                    print("Epoch {} - no calculated embedding".format(E))
            Pred_L1O[i] = np.array(Pred_L1O[i])
            valid_epochs[i] = np.array(valid_epochs[i])

        combined_epochs = merge_epochs(valid_epochs)
        P, P_std = mean_cross_validated_index(Pred_L1O, valid_epochs, combined_epochs)

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
            P, P_std = np.mean(Pred_L1O, axis=-1), np.std(Pred_L1O, axis=-1)

    return P, P_std, combined_epochs


def eval_retrieval(run, net_type, metric, epochs, dset, NN=[7, 11, 17], cross_validation=False, n_groups=5):
    Embed = FileManager.Embed(net_type)
    Prec, Prec_b, Prec_m = [[] for i in range(n_groups)], [[] for i in range(n_groups)], [[] for i in range(n_groups)]
    valid_epochs = [[] for i in range(n_groups)]
    if cross_validation:
        # Load
        embed_source = [Embed(run + 'c{}'.format(c), dset) for c in range(n_groups)]
        Ret = Retriever(title='{}-{}'.format(net_type, run), dset=dset)
        for i, source in enumerate(embed_source):
            Ret.load_embedding(source, multi_epcch=True)
            for E in epochs:
                # Calc
                prec, prec_b, prec_m = [], [], []
                try:
                    Ret.fit(np.max(NN), metric=metric, epoch=E)
                except:
                    print("Epoch {} - no calculated embedding".format(E))
                    continue
                for N in NN:
                    p, pb, pm = Ret.evaluate_precision(n=N)
                    prec.append(p)
                    prec_b.append(pb)
                    prec_m.append(pm)
                Prec[i].append(np.array(prec))
                Prec_b[i].append(np.array(prec_b))
                Prec_m[i].append(np.array(prec_m))
                valid_epochs[i].append(E)

            Prec[i] = np.array(Prec[i])
            Prec_b[i] = np.array(Prec_b[i])
            Prec_m[i] = np.array(Prec_m[i])
            valid_epochs[i] = np.array(valid_epochs[i])

        combined_epochs = epochs  # merge_epochs(valid_epochs)
        P, P_std = mean_cross_validated_index(Prec, valid_epochs, combined_epochs)
        #P, P_std = np.mean(np.mean(Prec, axis=-1), axis=0), np.mean(np.std(Prec, axis=-1), axis=0)
        combined = 2 * np.array(Prec_b) * np.array(Prec_m) / (np.array(Prec_b) + np.array(Prec_m))
        #F1, F1_std = np.mean(np.mean(combined, axis=-1), axis=0), np.mean(np.std(combined, axis=-1), axis=0)
        F1, F1_std = mean_cross_validated_index(combined, valid_epochs, combined_epochs)

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

        Prec = np.array(Prec)
        Prec_m = np.array(Prec_m)
        Prec_b = np.array(Prec_b)
        f1 = 2 * Prec_b * Prec_m / (Prec_b + Prec_m)
        P, P_std = np.mean(Prec, axis=-1), np.std(Prec, axis=-1)
        F1, F1_std = np.mean(f1, axis=-1), np.std(f1, axis=-1)

    return P, P_std, F1, F1_std, valid_epochs


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
            except:
                print("Epoch {} - no calculated embedding".format(E))
                continue
            Reg.evaluate_rating_space(norm=rating_norm)
            Reg.evaluate_rating_distance_matrix(method=metric, clustered_rating_distance=False)

            pm, _, km = Reg.correlate_retrieval('embed', 'malig')
            pr, _, kr = Reg.correlate_retrieval('embed', 'rating')
            valid_epochs.append(E)

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