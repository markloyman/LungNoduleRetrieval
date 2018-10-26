import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from LIDC.lidcUtils import calc_rating
from Network.dataUtils import rating_normalize
from Network.data_loader import load_nodule_raw_dataset


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

    def load_embedding(self, filename, multi_epcch=False):
        self.multi_epcch = multi_epcch
        if type(filename) is not list:
            filename = [filename]
        self.images, self.embedding, self.meta_data, self.labels, self.masks, self.rating = [], [], [], [], [], []
        for fn in filename:
            try:
                assert(type(fn) is str)
                if multi_epcch:
                    embedding, epochs, meta_data, images, classes, labels, masks = pickle.load(open(fn, 'br'))
                    epochs = np.array(epochs)
                    embed_concat_axis = 1
                else:
                    images, embedding, meta_data, labels, masks = pickle.load(open(fn, 'br'))
                    epochs = None
                    embed_concat_axis = 0
                    classes = labels
                self.images.append(images)
                self.embedding.append(embedding)
                self.meta_data += meta_data
                self.labels.append(classes)
                self.rating.append(labels)
                self.masks.append(masks)
                self.epochs = epochs
            except:
                print("failed to load " + fn)
        assert len(self.images) > 0
        self.images = np.concatenate(self.images)
        self.embedding = np.concatenate(self.embedding, axis=embed_concat_axis)
        self.labels = np.concatenate(self.labels)
        self.rating = np.concatenate(self.rating)
        self.masks = np.concatenate(self.masks)

        if self.labels.shape[1] > 1:
            self.labels = np.argmax(self.labels, axis=1)

        self.len = len(self.meta_data)
        self.nod_ids = [None]*self.len

        self.images = np.squeeze(self.images)
        self.labels = np.squeeze(self.labels)

        print("Loaded {} entries from {}".format(self.len, filename))

        return self.embedding, self.epochs

    def load_rating(self, dataset):
        #images, mask, rating, meta, labels, nod_ids = \
        #   zip(*[ (e['patch'], e['mask'], e['rating'], e['info'], e['label'], e['nod_ids'] ) for e in dataset])
        images, mask, labels, meta, size, rating, weights, _ = zip(*dataset)

        images = [im*(1.0-0.5*ms) for im, ms in zip(images, mask)]
        images = np.squeeze(images)
        labels = np.squeeze(labels)
        rating = [ np.mean(r, axis=(0)) for r in rating]

        self.images, self.embedding, self.meta_data, self.labels, self.nod_ids = images, rating, meta, labels, meta
        self.len = len(self.meta_data)
        self.multi_epcch = False

        print("Loaded {} entries from dataset".format(self.len))

    def fit(self, n=None, metric='l2', normalization='None', epoch=None, label=None):
        self.n = n

        if self.multi_epcch:
            assert(epoch is not None)
            assert (self.epochs is not None)
            epoch_idx = np.argwhere(epoch == self.epochs)[0][0]
            embedding = self.embedding[epoch_idx]
        else:
            embedding = self.embedding

        if label is not None:
            embedding = embedding[self.labels == label]

        if self.n is None:
            self.n = embedding.shape[0] - 1

        nbrs = NearestNeighbors(n_neighbors=(self.n+1), algorithm='auto', metric=metric).fit(rating_normalize(embedding, normalization))
        distances, indices = nbrs.kneighbors(rating_normalize(embedding, normalization))
        self.indices = indices[:, 1:]
        self.distances = distances[:, 1:]

    def ret_nbrs(self, n_top=None):
        if n_top is None:
            n_top = self.n
        assert n_top <= self.n

        return self.indices[:,:n_top], self.distances[:,:n_top]

    def ret(self, query, n_top=None, return_distance=False):
        if n_top is None:
            n_top = self.n
        assert n_top <= self.n

        nn = self.indices[query, :n_top]
        if return_distance:
            dd = self.distances[query, :n_top]
            return nn, dd
        else:
            return nn

    def show(self, title, image, label, rating=None, meta=None, plot_handle=None, nods=None):
        L = 'BMU'
        #ann = getAnnotation(meta)
        #ann.visualize_in_scan()
        #if nods is None:
        #    feature_vals = calc_rating(meta, method='single')
        #    print('single -> {}'.format(feature_vals))
        #else:
        feature_vals = calc_rating(meta, nodule_ids=nods) if rating is None else np.mean(rating, axis=0)
        print('mean ->: {}'.format(feature_vals))

        if plot_handle is None:
            plt.figure()
            plot_handle = plt.subplot(111)
        plot_handle.axes.title.set_text("{} - {} ({})".format(title, L[label], np.round(feature_vals).astype('int')))
        plot_handle.imshow(image, cmap='gray')

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

    def compare_ret(self, query, n_top=None, method='mean'):
        if n_top == None: n_top = self.n
        assert n_top <= self.n

        nn, dd = self.ret(query, n_top=n_top, return_distance=True)
        print("N: {}".format(nn))

        plt.figure(self.title + ' Retrieval #' + str(query))

        handle = plt.subplot(2, 3, 1)
        self.show('Query', self.images[query], self.labels[query], rating=self.rating[query], plot_handle=handle, nods=self.nod_ids[query])

        l2 = lambda x, y: np.sqrt((x-y).dot(x-y))
        qr = np.mean(self.rating[query], axis=0)
        dists = np.array([l2(qr, np.mean(r, axis=0)) for r in self.rating])

        handle = plt.subplot(2, 3, 2)
        handle.hist(dists.flatten(), bins=20)

        for idx in range(2*3 - 2):
            handle = plt.subplot(2,3,idx+3)
            self.show('Ret#{} [d{:.2f}]'.format(idx, dists[nn[idx]]), self.images[nn[idx]], self.labels[nn[idx]], rating=self.rating[nn[idx]], plot_handle=handle, nods=self.nod_ids[nn[idx]])



    def classify_naive(self):
        clf = KNeighborsClassifier(self.n, weights='uniform')
        clf.fit(self.embedding, self.labels)

        pred = clf.predict(self.embedding)

        return pred

    def classify_leave1out(self, epoch=None, n=None, metric='l2', verbose=False):
        if self.multi_epcch:
            assert(epoch is not None)
            assert (self.epochs is not None)
            epoch_idx = np.argwhere(epoch == self.epochs)[0][0]
            embedding = self.embedding[epoch_idx]
        else:
            embedding = self.embedding
        if n is None:
            n = self.n

        model_select = LeaveOneOut()
        clf = KNeighborsClassifier(n, weights='uniform', metric=metric)
        pred = np.zeros((len(self.labels), 1))
        for train_index, test_index in model_select.split(embedding):
            clf.fit(embedding[train_index], self.labels[train_index])
            pred[test_index] = clf.predict(embedding[test_index])
        acc = accuracy(self.labels, pred)
        if verbose:
            print('Classification Accuracy: {:.2f}'.format(acc))
        return (pred, self.labels), acc

    def classify_kfold(self, epoch=None, n=None, metric='l2', verbose=False, k_fold=None):
        if self.multi_epcch:
            assert(epoch is not None)
            assert (self.epochs is not None)
            epoch_idx = np.argwhere(epoch == self.epochs)[0][0]
            embedding = self.embedding[epoch_idx]
        else:
            embedding = self.embedding
        if n is None:
            n = self.n

        model_select = KFold(n_splits=k_fold, shuffle=False)
        clf = KNeighborsClassifier(n, weights='uniform', metric=metric)
        acc = []
        for train_index, test_index in model_select.split(embedding):
            clf.fit(embedding[train_index], self.labels[train_index])
            pred = clf.predict(embedding[test_index])
            acc += [accuracy(self.labels[test_index], pred)]
        if verbose:
            print('Classification Accuracy: {:.2f}'.format(acc))
        return np.mean(acc)

    def evaluate_precision(self, n=None):
        Acc = [[], []]
        for idx in range(self.len):
            nn = self.ret(idx, n_top=n)
            true = self.labels[idx]
            acc = accuracy(true, self.labels[nn])
            Acc[true].append(acc)
        precision_benign = np.mean(Acc[0])
        precision_malig = np.mean(Acc[1])
        precision_total = np.mean(np.concatenate(Acc))

        return precision_total, precision_benign, precision_malig

    def pca(self, epoch = None, plt_=None, label=None):
        #Metric = DistanceMetric.get_metric(metric)
        #DM = Metric.pairwise(self.embedding)
        epoch_idx = np.argwhere(epoch == self.epochs)[0][0]
        embed = self.embedding if epoch is None else self.embedding[epoch_idx]
        E = PCA(n_components=2).fit_transform(embed)

        #plt.figure()
        leg = []
        if label is None or label == 0:
            plt_.scatter(E[self.labels == 0, 0], E[self.labels == 0, 1], c='blue', s=1, alpha=0.2)
            leg += ['B']
        if label is None or label == 1:
            plt_.scatter(E[self.labels == 1, 0], E[self.labels == 1, 1], c='red', s=1, alpha=0.2)
            leg += ['M']
        plt_.legend(leg)
        plt_.axes.title.set_text('PCA: {}-{}, {}'.format(self.title, epoch, self.set))
        #plt_.title('PCA: {}, {}'.format(self.title, self.set))


# -----------------------------------
#           __main__
# -----------------------------------

if __name__ == "__main__":

    from Network import FileManager

    dset = 'Valid'

    wRuns = ['512cc0', '251c0']  #['064X', '078X', '026'] #['064X', '071' (is actually 071X), '078X', '081', '082']
    wRunsNet = ['dirRS', 'dirR']  #, 'dir']
    run_metrics = ['l2']

    select = 1

    rating_normalizaion = 'None' # 'None', 'Normal', 'Scale'

    doRatingRet     = True
    doPCA           = False


    if doRatingRet:
        Embed = FileManager.Embed(wRunsNet[select])
        #N = 5
        #testData, validData, trainData = load_nodule_raw_dataset(size=160, res=0.5, sample='Normal')
        ##if dset is 'Train':  data = trainData
        #if dset is 'Test':   data = testData
        #if dset is 'Valid':  data = validData

        #Ret = Retriever(title='Ratings', dset=set)
        #Ret.load_rating(data)
        #et.fit(N)

        #info, nod_ids = Ret.show_ret(15)
        #info, nod_ids = Ret.show_ret(135)
        #info, nod_ids = Ret.show_ret(135)
        #anns = getAnnotation(info, nodule_ids=nod_ids, return_all=True)
        #pickle.dump(anns, open('tmp.p', 'bw'))

        Ret = Retriever(title=wRuns[select]+wRunsNet[select], dset=dset)
        Ret.load_embedding(Embed(wRuns[select], dset), multi_epcch=True)
        #Ret.fit(N, epoch=70)
        Ret.fit(epoch=70)

        #Ret.compare_ret(322)
        #Ret.compare_ret(153)
        #Ret.compare_ret(145)
        Ret.compare_ret(139)

        #Ret.show_ret(237)
        #Ret.show_ret(295)
        #Ret.show_ret(262)
        #Ret.show_ret(315)


    if doPCA:
        for run, net_type in zip(wRuns, wRunsNet):
            Embed = FileManager.Embed(net_type)
            Ret = Retriever(title=run, dset=dset)
            Ret.load_embedding(Embed(run, 40, dset))
            Ret.pca()

    plt.show()



