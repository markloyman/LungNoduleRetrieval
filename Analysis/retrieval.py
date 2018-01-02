import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from LIDC.lidcUtils import calc_rating
from Network.data import load_nodule_raw_dataset
from Network.dataUtils import rating_normalize

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


class Retriever:

    def __init__(self, title='', dset=''):
        self.title  = title
        self.set    = dset

        self.nod_ids = None

    def load_embedding(self, filename):

        self.images, self.embedding, self.meta_data, self.labels, self.masks \
            = pickle.load(open(filename, 'br'))
            #= pickle.load(open('./output/embed/{}'.format(filename), 'br'))
        self.len = len(self.meta_data)
        self.nod_ids = [None]*self.len

        self.images = np.squeeze(self.images)
        self.labels = np.squeeze(self.labels)

        print("Loaded {} entries from {}".format(self.len, filename))

        return self.embedding

    def load_rating(self, dataset):
        images, mask, rating, meta, labels, nod_ids = \
            zip(*[ (e['patch'], e['mask'], e['rating'], e['info'], e['label'], e['nod_ids'] ) for e in dataset])

        images = [im*(1.0-0.5*ms) for im, ms in zip(images, mask)]
        images = np.squeeze(images)
        labels = np.squeeze(labels)
        rating = [ np.mean(r, axis=(0)) for r in rating]

        self.images, self.embedding, self.meta_data, self.labels, self.nod_ids = images, rating, meta, labels, nod_ids
        self.len = len(self.meta_data)

        print("Loaded {} entries from dataset".format(self.len))

    def fit(self, n=None, metric='l1', normalization='None'):
        if n is None:
            self.n = self.embedding.shape[0]-1
        else:
            self.n = n

        nbrs = NearestNeighbors(n_neighbors=(self.n+1), algorithm='auto', metric=metric).fit(rating_normalize(self.embedding, normalization))
        distances, indices = nbrs.kneighbors(rating_normalize(self.embedding, normalization))
        self.indices = indices[:, 1:]
        self.distances = distances[:, 1:]

    def ret_nbrs(self, n_top=None):
        if n_top == None:
            n_top = self.n
        assert n_top <= self.n

        return self.indices[:,:n_top], self.distances[:,:n_top]

    def ret(self, query, n_top=None, return_distance=False):
        if n_top==None: n_top = self.n
        assert n_top <= self.n

        nn = self.indices[query, :n_top]
        if return_distance:
            dd = self.distances[query, :n_top]
            return nn, dd
        else:
            return nn

    def show(self, title, image, label, meta, nods=None):
        L = 'BM'
        #ann = getAnnotation(meta)
        #ann.visualize_in_scan()
        #if nods is None:
        #    feature_vals = calc_rating(meta, method='single')
        #    print('single -> {}'.format(feature_vals))
        #else:
        feature_vals = calc_rating(meta, nodule_ids=nods)
        print('mean ->: {}'.format(feature_vals))

        plt.figure()
        plt.title("{} - {} ({})".format(title, L[label], np.round(feature_vals, 1)))
        plt.imshow(image, cmap='gray')

    def show_ret(self, query, n_top=None, method='mean'):
        if n_top==None: n_top = self.n
        assert n_top <= self.n

        nn, dd = self.ret(query, n_top=n_top, return_distance=True)
        print("N: {}".format(nn))

        print([self.meta_data[n] for n in nn])
        print([calc_rating(self.meta_data[n], nodule_ids=self.nod_ids[n], method=method) for n in nn])

        self.show('Query', self.images[query], self.labels[query], self.meta_data[query], self.nod_ids[query])
        for idx in range(n_top):
            self.show('Ret#{} [d{:.2f}]'.format(idx, dd[idx]), self.images[nn[idx]], self.labels[nn[idx]], self.meta_data[nn[idx]], self.nod_ids[nn[idx]])

        if self.nod_ids[query] is not None:
            return self.meta_data[query], self.nod_ids[query]
        else:
            return self.meta_data[query]


    def classify_naive(self):
        clf = KNeighborsClassifier(self.n, weights='uniform')
        clf.fit(self.embedding, self.labels)

        pred = clf.predict(self.embedding)

        return pred

    def classify_leave1out(self):
        loo = LeaveOneOut()
        clf = KNeighborsClassifier(self.n, weights='uniform')
        pred = np.zeros((len(self.labels),1))
        for train_index, test_index in loo.split(self.embedding):
            clf.fit(self.embedding[train_index], self.labels[train_index])
            pred[test_index] = clf.predict(self.embedding[test_index])
        acc = accuracy(self.labels, pred)
        print('Classification Accuracy: {:.2f}'.format(acc))
        return (pred, self.labels), acc

    def evaluate_precision(self, pred = None, plot=False, split=False):
        #if pred is None:
        #    pred = self.classify_leave1out()[0]
        acc = []
        acc.append([])
        if split:
            acc.append([])

        set = 0
        for idx in range(self.len):
            nn = self.ret(idx)

            if split:
                set = self.labels[idx]

            acc[set].append(accuracy(self.labels[idx], self.labels[nn]))

        if split:
            prec = ( np.mean(acc[0]), np.mean(acc[1]) )
        else:
            prec = np.mean(acc[0])

        if plot:
            plt.figure()
            if split:
                plt.title("Precision")
                plt.subplot(121)
                plt.hist(acc[0])
                plt.title('Benign (Mean:{}), {}, {}'.format(prec[0], self.title, self.set))
                plt.subplot(122)
                plt.hist(acc[1])
                plt.title('Malignant (Mean:{}), {}, {}'.format(prec[1], self.title, self.set))
            else:
                plt.title('Precision (Mean:{}), {}, {}'.format(prec,self.title,self.set))
                plt.hist(acc[0])

        return prec

    def pca(self):
        #Metric = DistanceMetric.get_metric(metric)
        #DM = Metric.pairwise(self.embedding)
        E = PCA(n_components=2).fit_transform(self.embedding)

        plt.figure()
        plt.scatter(E[self.labels == 0,0], E[self.labels == 0,1], c='blue')
        plt.scatter(E[self.labels == 1,0], E[self.labels == 1,1], c='red')
        plt.legend(('B', 'M'))
        plt.title('PCA: {}, {}'.format(self.title, self.set))


# -----------------------------------
#           __main__
# -----------------------------------

if __name__ == "__main__":

    #WW = ['embed_siam000-15_Test.p', 'embed_siam000-25_Test.p', 'embed_siam001-30_Test.p', 'embed_siam001-40_Test.p']
    #leg = ['Chained-E15', 'Chained-E25', 'Base-E30', 'Base-E40']
    #WW = ['embed_siam000-10_Test.p', 'embed_siam000-15_Test.p', 'embed_siam000-20_Test.p', 'embed_siam000-25_Test.p', 'embed_siam000-30_Test.p',
    #      'embed_siam000-35_Test.p', 'embed_siam000-40_Test.p', 'embed_siam000-45_Test.p', 'embed_siam000-50_Test.p']
    from Analysis.RatingCorrelator import calc_distance_matrix
    import Analysis.metric_space_indexes as index
    import FileManager
    #Embed = FileManager.Embed('siam')

    dset = 'Train'

    wRuns = ['101', '103']  #['064X', '078X', '026'] #['064X', '071' (is actually 071X), '078X', '081', '082']
    wRunsNet = ['siam', 'siam']  #, 'dir']
    run_metrics = ['l1', 'l2']

    #wRuns = ['103']
    #wRunsNet = ['dir']  # , 'dir']
    #run_metrics = ['l2']

    rating_normalizaion = 'Normal' # 'None', 'Normal', 'Scale'

    wEpchs = [30, 35, 40, 45] #, 20, 25, 30, 35, 40, 45]
    #WW = ['embed_siam{}-{}_{}.p'.format(run, E, dset) for E in wEpchs]
    #WW = [ Embed(run, E, dset) for E in wEpchs]

    leg = ['E{}'.format(E) for E in wEpchs]

    doClass         = False
    doRet           = False
    doRatingRet     = True
    doMetricSpaceIndexes = False
    doPCA           = False

    #Ret = Retriever(WW[-4], atitle='chn', aset=set)
    #Ret.fit(3)
    #Ret.show_ret(10)

    ##  ------------------------
    #       Classification
    ##  ------------------------

    if doClass:
        plt.figure('KNN Classification - ' + dset)
        for run, net_type, idx, metric in zip(wRuns, wRunsNet, range(len(wRuns)), run_metrics):
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
            plt.subplot(1, len(wRuns), idx+1)
            plt.plot(wEpchs,Pred_L1O, '-*')
            plt.grid(which='major', axis='y')
            plt.title('{}-{}'.format(net_type, run))
            plt.ylabel('ACC')
            plt.xlabel('epoch')
            plt.legend(NN)

        print('Done Classification.')

    ##  ------------------------
    #       Retrieval
    ##  ------------------------

    if doRet:

        NN = [3, 5, 7, 11, 17]

        for run, net_type, idx, metric in zip(wRuns, wRunsNet, range(len(wRuns)), run_metrics):
            NN = [3, 5, 7, 11, 17]
            Embed = FileManager.Embed(net_type)
            Prec, Prec_b, Prec_m = [], [], []
            WW = [Embed(run, E, dset) for E in wEpchs]

            for W in WW:
                Ret = Retriever(title='', dset='')
                Ret.load_embedding(W)

                prec, prec_b, prec_m = [], [], []
                for N in NN:
                    Ret.fit(N,  metric=metric)
                    p =         Ret.evaluate_precision(plot=False, split=False)
                    pm, pb =    Ret.evaluate_precision(plot=False, split=True)
                    prec.append(p)
                    prec_b.append(pb)
                    prec_m.append(pm)

                Prec.append(np.array(prec))
                Prec_b.append(np.array(prec_b))
                Prec_m.append(np.array(prec_m))

            #Pred_L1O = np.transpose(np.array(Pred_L1O))
            Prec   = (np.array(Prec))
            Prec_m = (np.array(Prec_m))
            Prec_b = (np.array(Prec_b))

            plt.figure('RET_'+run+'_'+dset)

            plt.subplot(211)
            plt.plot(wEpchs, Prec, '-*')
            plt.legend(NN)
            plt.title('Retrieval')
            plt.grid(which='major', axis='y')
            plt.ylim([0.7, 0.9])
            plt.ylabel('Precision')
            plt.xlabel('epoch')

            f1 = 2*Prec_b*Prec_m / (Prec_b+Prec_m)
            plt.subplot(212)
            plt.plot(wEpchs, f1, '-*')
            plt.legend(NN)
            plt.title('F1')
            plt.grid(which='major', axis='y')
            plt.ylim([0.7, 0.9])
            plt.ylabel('Retrieval Index')
            plt.xlabel('epoch')

            #plt.subplot(325)
            #plt.grid(which='major', axis='y')
            #plt.plot(wEpchs, Prec_b, '-*')
            #plt.legend(NN)
            #plt.title('Benign')
            #plt.ylim([0.6, 1.0])
            #plt.ylabel('Precision')
            #plt.xlabel('epoch')

            #plt.subplot(326)
            #plt.plot(wEpchs, Prec_m, '-*')
            #plt.legend(NN)
            #plt.title('Malignant')
            #plt.ylim([0.6, 1.0])
            #plt.grid(which='major', axis='y')
            #plt.ylabel('Precision')
            #plt.xlabel('epoch')

        print('Done Retrieval.')

    if doRatingRet:
        Embed = FileManager.Embed('siam')
        N = 5
        testData, validData, trainData = load_nodule_raw_dataset(size=144, res='0.5I', sample='Normal')
        if dset is 'Train':  data = trainData
        if dset is 'Test':   data = testData
        if dset is 'Valid':  data = validData

        #Ret = Retriever(title='Ratings', dset=set)
        #Ret.load_rating(data)
        #et.fit(N)

        #info, nod_ids = Ret.show_ret(15)
        #info, nod_ids = Ret.show_ret(135)
        #info, nod_ids = Ret.show_ret(135)
        #anns = getAnnotation(info, nodule_ids=nod_ids, return_all=True)
        #pickle.dump(anns, open('tmp.p', 'bw'))

        Ret = Retriever(title='', dset=dset)
        Ret.load_embedding(Embed('100', 40, dset))
        Ret.fit(N)

        Ret.show_ret(322)
        Ret.show_ret(153)
        Ret.show_ret(745)
        Ret.show_ret(339)

        Ret.show_ret(737)
        Ret.show_ret(295)
        Ret.show_ret(262)
        Ret.show_ret(315)

    if doMetricSpaceIndexes:
        metrics = ['l2']
        plt.figure()
        p = [None]*len(metrics)*5
        for i in range(5*len(metrics)):
            p[i] = plt.subplot(len(metrics), 5, i + 1)
        for m, metric in enumerate(metrics):
            print("Begin: {} metric".format(metric))
            for run, net_type, r in zip(wRuns, wRunsNet, range(len(wRuns))):
                Embed = FileManager.Embed(net_type)
                WW = [Embed(run, E, dset) for E in wEpchs]
                # init
                idx_hubness = np.zeros(len(WW))
                idx_hubness_std = np.zeros(len(WW))
                idx_symmetry = np.zeros(len(WW))
                idx_symmetry_std = np.zeros(len(WW))
                idx_concentration = np.zeros(len(WW))
                idx_concentration_std = np.zeros(len(WW))
                idx_contrast = np.zeros(len(WW))
                idx_contrast_std = np.zeros(len(WW))
                idx_kummar = np.zeros(len(WW))
                # calculate
                for e, W in enumerate(WW):
                    Ret = Retriever(title='{}'.format(run), dset=dset)
                    embd = Ret.load_embedding(W)
                    Ret.fit(metric=metric)
                    indices, distances = Ret.ret_nbrs()
                    distance_matrix = calc_distance_matrix(embd, metric)
                    # hubness
                    K = [3, 5, 7, 11, 17]
                    h = np.zeros(len(K))
                    #plt.figure()
                    for i in range(len(K)):
                        h_ = index.hubness(indices, K[i])
                        h[i] = h_[0]
                        if K[i] == 3:
                            j = 1
                        elif K[i] == 7:
                            j = 2
                        elif K[i] == 11:
                            j = 3
                        else:
                            j = 0
                        if False: #j != 0:
                            plt.subplot(len(metrics), 3, m * 3 + j)
                            plt.title('k-occ: {}'.format(net_type))
                            plt.ylabel('k={}'.format(K[i]))
                            plt.plot(np.array(range(len(h_[1]))), h_[1])
                    idx_hubness[e] = np.mean(h)
                    idx_hubness_std[e] = np.std(h)
                    #   symmetry
                    K = [3, 5, 7, 11, 17]
                    s = np.zeros(len(K))
                    for i in range(len(K)):
                        s[i] = index.symmetry(indices, K[i])
                    idx_symmetry[e] = np.mean(s)
                    idx_symmetry_std[e] = np.std(s)
                    # kumar index
                    tau, l_e = index.kumar(distances, res=0.0001)
                    idx_kummar[e] = tau
                    idx_concentration[e] = index.concentration(distance_matrix)[0]
                    idx_concentration_std[e] = index.concentration(distance_matrix)[1]
                    idx_contrast[e] = index.relative_contrast_imp(distance_matrix)[0]
                    idx_contrast_std[e] = index.relative_contrast_imp(distance_matrix)[1]
                # plot
                #   hubness
                q = p[5 * m + 0].plot(wEpchs, idx_hubness)
                p[5 * m + 0].plot(wEpchs, idx_hubness + idx_hubness_std, color=q[0].get_color(), ls='--')
                p[5 * m + 0].plot(wEpchs, idx_hubness - idx_hubness_std, color=q[0].get_color(), ls='--')
                #   symmetry
                q = p[5 * m + 1].plot(wEpchs, idx_symmetry)
                p[5 * m + 1].plot(wEpchs, idx_symmetry + idx_symmetry_std, color=q[0].get_color(), ls='--')
                p[5 * m + 1].plot(wEpchs, idx_symmetry - idx_symmetry_std, color=q[0].get_color(), ls='--')
                #   contrast
                q = p[5 * m + 2].plot(wEpchs, idx_contrast)
                p[5 * m + 2].plot(wEpchs, idx_contrast + idx_contrast_std, color=q[0].get_color(), ls='--')
                p[5 * m + 2].plot(wEpchs, idx_contrast - idx_contrast_std, color=q[0].get_color(), ls='--')
                #   concentration
                q = p[5 * m + 3].plot(wEpchs, idx_concentration)
                p[5 * m + 3].plot(wEpchs, idx_concentration + idx_concentration_std, color=q[0].get_color(), ls='--')
                p[5 * m + 3].plot(wEpchs, idx_concentration - idx_concentration_std, color=q[0].get_color(), ls='--')
                #   kumar
                p[5 * m + 4].plot(wEpchs, idx_kummar)
                # labels
                if r == 0: #first column
                    p[5 * m + 0].axes.yaxis.label.set_text(metric)
                if m == 0: #first row
                    p[5 * m + 0].axes.title.set_text('hubness')
                    p[5 * m + 1].axes.title.set_text('symmetry')
                    p[5 * m + 2].axes.title.set_text('contrast')
                    p[5 * m + 3].axes.title.set_text('concentration')
                    p[5 * m + 4].axes.title.set_text('kumari')
                if m == len(metrics)-1:  # last row
                    p[5 * m + 2].axes.xaxis.label.set_text('epochs')
        p[-1].legend(wRunsNet)
        print('Done doMetricSpaceIndexes')

    if doPCA:
        for run, net_type in zip(wRuns, wRunsNet):
            Embed = FileManager.Embed(net_type)
            Ret = Retriever(title=run, dset=dset)
            Ret.load_embedding(Embed(run, 40, dset))
            Ret.pca()

    plt.show()



