import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import DistanceMetric
from sklearn.decomposition import PCA
import pickle

from LIDC.lidcUtils import getAnnotation, calc_rating

def accuracy(true, pred):
    pred = np.clip(pred, 0, 1)
    pred = np.squeeze(np.round(pred).astype('uint'))
    mask = (true==pred).astype('uint')
    acc = np.mean(mask)
    return acc


def precision(query, nn):
    mask = (query == nn).astype('uint')
    np.sum(mask)


class Retriever:

    def __init__(self, filename, atitle='', aset=''):

        self.title = atitle
        self.set = aset

        self.images, self.embedding, self.meta_data, self.labels = pickle.load(open('./embed/{}'.format(filename),'br'))
        self.len = len(self.meta_data)

        self.images = np.squeeze(self.images)
        self.labels = np.squeeze(self.labels)

        print("Loaded {} entries from {}".format(self.len, filename))

    def fit(self, n):
        self.n = n
        nbrs = NearestNeighbors(n_neighbors=(n + 1), algorithm='auto').fit(self.embedding)

        distances, indices = nbrs.kneighbors(self.embedding)
        self.indices = indices[:, 1:]
        self.distances = distances[:, 1:]

    def fit_rating(self, n):
        self.n = n
        space = [calc_rating(meta) for meta in self.meta_data]
        nbrs = NearestNeighbors(n_neighbors=(n + 1), algorithm='brute', metric='l1').fit(space)

        distances, indices = nbrs.kneighbors(space)
        self.indices = indices[:, 1:]
        self.distances = distances[:, 1:]

    def ret(self, query, n_top=None, return_distance=False):
        if n_top==None: n_top = self.n
        assert n_top <= self.n

        nn = self.indices[query, :n_top]
        if return_distance:
            dd = self.distances[query, :n_top]
            return nn, dd
        else:
            return nn

    def show(self, title, image, label, meta):
        L = 'BM'
        #ann = getAnnotation(meta)
        #ann.visualize_in_scan()
        feature_vals = calc_rating(meta)
        plt.figure()
        plt.title("{} - {} ({})".format(title, L[label], feature_vals))
        plt.imshow(image)

    def show_ret(self, query, n_top=None):
        if n_top==None: n_top = self.n
        assert n_top <= self.n

        nn, dd = self.ret(query, n_top=n_top, return_distance=True)

        self.show('Query', self.images[query], self.labels[query], self.meta_data[query])
        for idx in range(n_top):
            self.show('Ret#{} [d{:.2f}]'.format(idx,dd[idx]), self.images[nn[idx]], self.labels[nn[idx]], self.meta_data[nn[idx]])


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
        print('Classification Accuracy: {}'.format(acc))
        return pred, acc

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

    set = 'Train'
    wRuns  =  ['000', '001']
    run = wRuns[0]

    wEpchs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    WW = ['embed_siam{}-{}_{}.p'.format(run, E, set) for E in wEpchs]

    leg = ['E{}'.format(E) for E in wEpchs]

    doClass = False
    doRet   = False
    doRatingRet = True

    #Ret = Retriever(WW[-4], atitle='chn', aset=set)
    #Ret.fit(3)
    #Ret.show_ret(10)

    ##  ------------------------
    #       Classification
    ##  ------------------------

    if doClass:

        NN = [3, 5, 7, 9, 11, 13, 17, 21, 27]


        Pred_L1O = []
        for W in WW:
            Ret = Retriever(W, atitle='', aset='')

            pred_l1o = []
            for N in NN:
                Ret.fit(N)
                pred_l1o.append(Ret.classify_leave1out()[1])

            Pred_L1O.append(np.array(pred_l1o))

        #Pred_L1O = np.transpose(np.array(Pred_L1O))
        Pred_L1O = (np.array(Pred_L1O))
        plt.figure()
        plt.plot(wEpchs,Pred_L1O, '-*')
        plt.title('KNN Classification (Accuracy)')
        plt.legend(NN)


    ##  ------------------------
    #       Retrieval
    ##  ------------------------

    if doRet:

        NN = [3, 5, 13]

        Prec, Prec_b, Prec_m = [], [], []

        for W in WW:
            Ret = Retriever(W, atitle='', aset='')

            prec, prec_b, prec_m = [], [], []
            for N in NN:
                Ret.fit(N)
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

        plt.figure('Retrieval (Precision)')

        plt.subplot(311)
        plt.plot(wEpchs, Prec, '-*')
        plt.legend(NN)
        plt.title('Retrieval (Precision)')

        plt.subplot(312)
        plt.plot(wEpchs, Prec_b, '-*')
        plt.legend(NN)
        plt.title('Benign')

        plt.subplot(313)
        plt.plot(wEpchs, Prec_m, '-*')
        plt.legend(NN)
        plt.title('Malignant')


    if doRatingRet:
        N = 2
        Ret = Retriever(WW[-4], atitle='', aset='')
        
        Ret.fit_rating(N)
        Ret.show_ret(26)

        Ret.fit(N)
        Ret.show_ret(26)
            
    #Ret = Retriever(WW[-2], atitle='Epch40', aset='Test')
    #Ret.fit(n=7)
    #Ret.pca()

    plt.show()