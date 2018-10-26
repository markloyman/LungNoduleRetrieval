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
    merged = np.zeros((index[0].shape[0], len(combined_epochs)))
    for ep_id, epoch in enumerate(combined_epochs):
        #collect = [[idx[:, i] for i, e in enumerate(ve) if e == epoch] for idx, ve in zip(index, valid_epochs) if epoch in ve]
        collect = []
        for idx, ve in zip(index, valid_epochs):
            if epoch in ve:
                for i, e in enumerate(ve):
                    if e == epoch:
                        collect += [idx[:, i]]
        merged[:, ep_id] = np.mean(collect, axis=0)
    return merged


def mean_cross_validated_index_with_std(index, valid_epochs, combined_epochs):
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

        combined_epochs = merge_epochs(valid_epochs, min_element=max(n_groups-1, 1))
        P, P_std = mean_cross_validated_index_with_std(Pred_L1O, valid_epochs, combined_epochs)

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
        P, P_std = mean_cross_validated_index_with_std(Prec, valid_epochs, combined_epochs)
        #P, P_std = np.mean(np.mean(Prec, axis=-1), axis=0), np.mean(np.std(Prec, axis=-1), axis=0)
        combined = 2 * np.array(Prec_b) * np.array(Prec_m) / (np.array(Prec_b) + np.array(Prec_m))
        #F1, F1_std = np.mean(np.mean(combined, axis=-1), axis=0), np.mean(np.std(combined, axis=-1), axis=0)
        F1, F1_std = mean_cross_validated_index_with_std(combined, valid_epochs, combined_epochs)

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


def eval_correlation(run, net_type, metric, rating_metric, epochs, dset, objective='rating', rating_norm='none', cross_validation=False, n_groups=5, seq=False):

    Embed = FileManager.Embed(net_type)

    if cross_validation:
        # Load
        if n_groups > 1:
            embed_source = [Embed(run + 'c{}'.format(c), dset) for c in range(n_groups)]
        else:
            embed_source = [Embed(run + 'c{}'.format(c), dset) for c in [1]]

        valid_epochs = [[] for i in range(n_groups)]
        Pm, Km, Pr, Kr = [[] for i in range(n_groups)], [[] for i in range(n_groups)], [[] for i in range(n_groups)], [[] for i in range(n_groups)]
        PmStd, KmStd, PrStd, KrStd = [[] for i in range(n_groups)], [[] for i in range(n_groups)], [[] for i in range(n_groups)], [[] for i in range(n_groups)]

        for c_idx, source in enumerate(embed_source):
            Reg = RatingCorrelator(source, conf=c_idx, multi_epoch=True, seq=seq)

            # load rating data
            cache_filename = 'output/cached_{}_{}_{}.p'.format(objective, source.split('/')[-1][6:-2], c_idx)
            if not Reg.load_cached_rating_distance(cache_filename):
                print('evaluating rating distance matrix...')
                Reg.evaluate_rating_space(norm=rating_norm, ignore_labels=False)
                Reg.evaluate_rating_distance_matrix(method=rating_metric, clustered_rating_distance=True, weighted=True)
                Reg.dump_rating_distance_to_cache(cache_filename)
                #print('\tno dump for rating distance matrix...')

            if objective == 'size':
                print('evaluating size distance matrix...')
                Reg.evaluate_size_distance_matrix()

            for E in epochs:
                # Calc
                try:
                    Reg.evaluate_embed_distance_matrix(method=metric, epoch=E)
                except:
                    #print("Epoch {} - no calculated embedding".format(E))
                    continue

                pm, _, km = Reg.correlate_retrieval('embed', 'malig' if objective == 'rating' else 'size', verbose=False)
                pr, _, kr = Reg.correlate_retrieval('embed', 'rating', verbose=False)
                valid_epochs[c_idx].append(E)

                Pm[c_idx].append(pm[0])
                Km[c_idx].append(km[0])
                Pr[c_idx].append(pr[0])
                Kr[c_idx].append(kr[0])
                PmStd[c_idx].append(pm[1])
                KmStd[c_idx].append(km[1])
                PrStd[c_idx].append(pr[1])
                KrStd[c_idx].append(kr[1])

            Pm[c_idx] = np.expand_dims(Pm[c_idx], axis=0)
            Km[c_idx] = np.expand_dims(Km[c_idx], axis=0)
            Pr[c_idx] = np.expand_dims(Pr[c_idx], axis=0)
            Kr[c_idx] = np.expand_dims(Kr[c_idx], axis=0)
            PmStd[c_idx] = np.expand_dims(PmStd[c_idx], axis=0)
            KmStd[c_idx] = np.expand_dims(KmStd[c_idx], axis=0)
            PrStd[c_idx] = np.expand_dims(PrStd[c_idx], axis=0)
            KrStd[c_idx] = np.expand_dims(KrStd[c_idx], axis=0)

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

    merged_epochs = merge_epochs(valid_epochs, min_element=max(n_groups - 1, 1))
    Pm = mean_cross_validated_index(Pm, valid_epochs, merged_epochs)
    Km = mean_cross_validated_index(Km, valid_epochs, merged_epochs)
    Pr = mean_cross_validated_index(Pr, valid_epochs, merged_epochs)
    Kr = mean_cross_validated_index(Kr, valid_epochs, merged_epochs)
    PmStd = mean_cross_validated_index(PmStd, valid_epochs, merged_epochs)
    KmStd = mean_cross_validated_index(KmStd, valid_epochs, merged_epochs)
    PrStd = mean_cross_validated_index(PrStd, valid_epochs, merged_epochs)
    KrStd = mean_cross_validated_index(KrStd, valid_epochs, merged_epochs)

    return np.squeeze(Pm), np.squeeze(PmStd), np.squeeze(Km), np.squeeze(KmStd), np.squeeze(Pr), np.squeeze(PrStd), np.squeeze(Kr), np.squeeze(KrStd), np.array(merged_epochs)