import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from LIDC.lidcUtils import calc_rating
from Network.data_loader import load_nodule_raw_dataset
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
        if n_top == None: n_top = self.n
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

    dset = 'Valid'

    wRuns = ['011']  #['064X', '078X', '026'] #['064X', '071' (is actually 071X), '078X', '081', '082']
    wRunsNet = ['dirR']  #, 'dir']
    run_metrics = ['l2']

    #wRuns = ['103']
    #wRunsNet = ['dir']  # , 'dir']
    #run_metrics = ['l2']

    rating_normalizaion = 'Scale' # 'None', 'Normal', 'Scale'

    wEpchs = [10, 20, 30, 35, 40, 45, 50, 55] #, 20, 25, 30, 35, 40, 45]
    #WW = ['embed_siam{}-{}_{}.p'.format(run, E, dset) for E in wEpchs]
    #WW = [ Embed(run, E, dset) for E in wEpchs]

    leg = ['E{}'.format(E) for E in wEpchs]

    doClass         = False
    doRet           = False
    doRatingRet     = False
    doMetricSpaceIndexes = True
    doPCA           = False

    #Ret = Retriever(WW[-4], atitle='chn', aset=set)
    #Ret.fit(3)
    #Ret.show_ret(10)

    ##  ------------------------
    #       Classification
    ##  ------------------------


    ##  ------------------------
    #       Retrieval
    ##  ------------------------


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




    if doPCA:
        for run, net_type in zip(wRuns, wRunsNet):
            Embed = FileManager.Embed(net_type)
            Ret = Retriever(title=run, dset=dset)
            Ret.load_embedding(Embed(run, 40, dset))
            Ret.pca()

    plt.show()



