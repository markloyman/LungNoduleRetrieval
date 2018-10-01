import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau  # , wasserstein_distance
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from Analysis.analysis import calc_distance_matrix, calc_cross_distance_matrix
from LIDC.lidcUtils import calc_rating
from Network.dataUtils import rating_normalize
from scipy.spatial.distance import pdist, cdist, squareform

from Network.Common.utils import rating_clusters_distance_matrix

'''
def calc_distance_matrix(X, method):
    rating_mean = np.array(
        [3.75904203, 1.01148583, 5.50651678, 3.79985337, 3.96358749, 1.67269469, 1.56867058, 4.4591072, 2.55197133])
    rating_std.py = np.array(
        [1.09083287, 0.11373469, 1.02463593, 0.80119638, 1.04281277, 0.89359593, 0.89925905, 1.04813052, 1.12151403])
    rating_min = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1])
    rating_max = np.array(
        [5, 4, 6, 5, 5, 5, 5, 5, 5])
    if len(method) > 4 and method[-4:] == 'norm':
        X -= rating_mean
        X /= rating_std

    if len(method) > 4 and method[-4:] == 'scale':
        X = (X - rating_min)/(rating_max - rating_min)

    if method in ['chebyshev', 'euclidean', 'l1', 'l2']:
        DM = DistanceMetric.get_metric(method).pairwise(X)
    elif method in ['l1_norm', 'l2_norm', 'chebyshev_norm']:
        DM = DistanceMetric.get_metric(method[:-5]).pairwise(X)
    elif method in ['cosine', 'cosine_norm']:
        DM = pairwise.cosine_distances(X)
    elif method in ['correlation', 'cityblock', 'braycurtis', 'canberra', 'hamming', 'jaccard', 'kulsinski']:
        DM = squareform(pdist(X, method))
    elif method in ['emd']:
        l = len(X)
        DM = np.zeros((l, l))
        for x in range(l):
            for y in range(l):
                DM[x, y] = wasserstein_distance(X[x], X[y])
    else:
        return None

    return DM


def calc_cross_distance_matrix(X, Y, method):
    if method in ['chebyshev', 'euclidean', 'cosine', 'correlation', 'cityblock']:
        #DM = squareform(cdist(X, Y, method))
        DM = cdist(X, Y, method)
    else:
        return None

    return DM
'''

def flatten_dm(DM):
    return DM[np.triu_indices(DM.shape[0], 1)].reshape(-1, 1)


def linear_regression(x, y):
    regr = linear_model.LinearRegression()
    regr.fit(x,y)
    y_pred = regr.predict(x)

    return mean_squared_error(y, y_pred), r2_score(y, y_pred)


class RatingCorrelator:

    def __init__(self, filename, conf=None, title='', set='', multi_epoch=False):
        self.title  = title
        self.set    = set

        if multi_epoch: assert conf is not None

        self.load_embedding(filename, multi_epcch=multi_epoch, load_weights=True, confs=conf)
        self.len = len(self.meta_data)

        self.images = np.squeeze(self.images)
        self.labels = np.squeeze(self.labels)

        print("Loaded {} entries from {}".format(self.len, filename))

        # initialization
        self.embed_distance_matrix  = None
        self.rating_distance_matrix = None
        self.rating                 = None
        self.rating_metric          = ''
        self.embed_metric           = ''


    def load_embedding(self, source, confs, load_weights=False, multi_epcch=False):
        self.multi_epcch = multi_epcch
        if type(source) is not list:
            source = [source]
        if type(confs) is not list:
            confs = [confs]
        self.images, self.embedding, self.meta_data, self.labels, self.classes, self.masks, self.weights = [], [], [], [], [], [], []
        for c, fn in zip(confs, source):
            try:
                assert(type(fn) is str)
                if multi_epcch:
                    embedding, epochs, meta_data, images, classes, labels, masks = pickle.load(open(fn, 'br'))
                    if type(meta_data) is np.ndarray:
                        m = meta_data
                        meta_data = images
                        images = m
                    weights = None
                    if load_weights:
                        size = len(meta_data)
                        if size < 400:
                            data_type = 'Clean'
                        elif size < 700:
                            data_type = 'Primary'
                        else:
                            data_type = 'Full'
                        data_filename = './Dataset/Dataset{}CV{}_{:.0f}-{}-{}.p'.format(data_type, c, 160, 0.5, 'Normal')
                        data = pickle.load(open(data_filename, 'br'))
                        n = len(data)
                        assert n == len(meta_data)
                        assert np.all([data[i]['info'][0] == meta_data[i][0] for i in range(n)])
                        weights = [entry['weights'] for entry in data]

                    epochs = np.array(epochs)
                    embed_concat_axis = 1
                else:
                    images, embedding, meta_data, labels, masks = pickle.load(open(fn, 'br'))
                    classes = labels
                    weights = None
                    epochs = None
                    embed_concat_axis = 0
                self.images.append(images)
                self.embedding.append(embedding)
                self.meta_data += meta_data
                self.classes.append(classes)
                self.labels.append(labels)
                self.masks.append(masks)
                self.weights.append(weights)
                self.epochs = epochs
            except:
                print("failed to load " + fn)
        assert len(self.images) > 0
        self.images = np.concatenate(self.images)
        self.embedding = np.concatenate(self.embedding, axis=embed_concat_axis)
        self.labels = np.concatenate(self.labels)
        self.masks = np.concatenate(self.masks)
        self.weights = np.concatenate(self.weights)

    def load_cached_rating_distance(self, filename='output/cache_ratings.p'):
        #return False
        try:
            rating, dm, metric  = pickle.load(open(filename, 'br'))
            self.rating = rating
            self.rating_distance_matrix = dm
            self.rating_metric = metric
            print('loaded cached ratings data from: ' + filename)
            return True
        except:
            return False

    def dump_rating_distance_to_cache(self, filename='output/cache_ratings.p'):
        try:
            file = open(filename, 'bw')
        except:
            print('failed to dump rating data')
            return False

        pickle.dump((self.rating, self.rating_distance_matrix, self.rating_metric), file)
        return True

    def evaluate_rating_space(self, norm='none', ignore_labels=False):
        if np.concatenate(self.labels).ndim == 1 or ignore_labels:
            print('calc_from_meta')
            self.rating = [rating_normalize(calc_rating(meta, method='raw'), method=norm) for meta in self.meta_data]

        else:
            print('calc_from_labels')
            self.rating = [rating_normalize(lbl, method=norm) for lbl in self.labels]
        self.rating_distance_matrix = None  # reset after recalculating the ratings

    def evaluate_rating_distance_matrix(self, method='chebyshev', clustered_rating_distance=False, weighted=False):
        if clustered_rating_distance:
            n = len(self.rating)
            self.rating_distance_matrix = rating_clusters_distance_matrix(self.rating, weights=self.weights if weighted else None)
        else:
            rating = np.array([np.mean(rat, axis=0) for rat in self.rating])
            self.rating_distance_matrix = calc_distance_matrix(rating, method)
        #assert self.rating_distance_matrix.shape[0] == self.embed_distance_matrix.shape[0]
        self.rating_metric = method
        return self.rating_distance_matrix

    def evaluate_embed_distance_matrix(self, method='euclidean', round=False, epoch=None):
        if self.multi_epcch:
            assert (epoch is not None)
            assert (self.epochs is not None)
            epoch_idx = np.argwhere(epoch == self.epochs)[0][0]
            embd = self.embedding[epoch_idx]
            assert np.all(np.isfinite(embd))
        else:
            embd = self.embedding
        embd = embd if (round==False) else np.round(embd)
        self.embed_distance_matrix = calc_distance_matrix(embd, method)
        self.embed_metric = method
        assert np.all(np.isfinite(self.embed_distance_matrix))

    def evaluate_size_distance_matrix(self):
        nodule_size = np.array([np.count_nonzero(q) for q in self.masks]).reshape(-1, 1) * 0.5 * 0.5
        tresh = [0, 15, 30, 60, 120]
        nodule_size = np.digitize(nodule_size, tresh)

        self.size_distance_matrix = calc_distance_matrix(nodule_size, 'l1')
        self.size_metric = 'l1'

        return self.size_distance_matrix

    def correlate_to_ratings(self, method='euclidean', round=False, epoch=None, epsilon=1e-9):
        if self.multi_epcch:
            assert (epoch is not None)
            assert (self.epochs is not None)
            epoch_idx = np.argwhere(epoch == self.epochs)[0][0]
            embd = self.embedding[epoch_idx]
            assert np.all(np.isfinite(embd))
        else:
            embd = self.embedding
        embd = embd if (round==False) else np.round(embd)
        assert np.all(np.isfinite(embd))

        rating = np.array([np.mean(r, axis=0) for r in self.rating])
        rating = rating if (round == False) else np.round(rating)

        params = rating.shape[1]
        assert params == embd.shape[1]
        corr = np.zeros(params)
        for p in range(params):
            corr[p] = pearsonr(embd[:, p] + epsilon, rating[:, p])[0]

        return corr

    def linear_regression(self):
        embed_dist = flatten_dm(self.embed_distance_matrix)
        rating_dist = flatten_dm(self.rating_distance_matrix)

        regr = linear_model.LinearRegression()
        regr.fit(embed_dist, rating_dist)

        # Make predictions using the testing set
        rating_dist_pred = regr.predict(embed_dist)

        # The coefficients
        print('Coefficients: \n', regr.coef_)
        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(rating_dist, rating_dist_pred))
        # Explained variance score: 1 is perfect prediction
        print('Variance score (r2): %.2f' % r2_score(rating_dist, rating_dist_pred))

    def load(self, name):
        if name is 'embed':
            xVec = self.embedding
        elif name is 'rating':
            xVec = np.array(self.rating)
        elif name is 'malig':
            xVec = [calc_rating(meta, 'malig') for meta in self.meta_data]
            xVec = np.array(xVec).reshape(-1, 1).astype('float64')
            xVec += 0.1*np.random.random([xVec.shape[0], xVec.shape[1]])
        else:
            assert False
        return xVec

    def load_distance_matrix(self, name, flat=True):
        if name == 'embed':
            xVec = self.embed_distance_matrix
            xMet = self.embed_metric
        elif name == 'rating':
            xVec = self.rating_distance_matrix
            xMet = self.rating_metric
        elif name == 'malig':
            #malig_rating = [calc_rating(meta, method='malig') for meta in self.meta_data]
            #malig_rating = np.array(malig_rating).reshape(-1, 1).astype('float64')
            malig_rating = np.array([[np.mean(rat[:, -1])] for rat in self.rating])
            xVec = calc_distance_matrix(malig_rating, method='euclidean')
            xMet = 'euclidean'
        elif name == 'size':
            xVec = self.size_distance_matrix
            xMet = self.size_metric
        else:
            assert False

        return flatten_dm(xVec) if flat else xVec, xMet

    def scatter(self, X, Y, xMethod = 'euclidean', yMethod = 'euclidean', sub=False):
        xVec = self.load(X)
        yVec = self.load(Y)

        maskMalig = self.labels == 1
        maskBenig = self.labels == 0

        x_benig_d = flatten_dm(calc_distance_matrix(        xVec[maskBenig], xMethod))
        x_malig_d = flatten_dm(calc_distance_matrix(        xVec[maskMalig], xMethod))
        x_cross_d = flatten_dm(calc_cross_distance_matrix(  xVec[maskMalig],
                                                            xVec[maskBenig], xMethod))

        y_benig_d = flatten_dm(calc_distance_matrix(        yVec[maskBenig], yMethod))
        y_malig_d = flatten_dm(calc_distance_matrix(        yVec[maskMalig], yMethod))
        y_cross_d = flatten_dm(calc_cross_distance_matrix(  yVec[maskMalig],
                                                            yVec[maskBenig], yMethod))

        print('{}-{}:'.format(X,Y))
        print('\tCross R2 = {}'.format(linear_regression(x_cross_d, y_cross_d)[1]))
        print('\tMalig R2 = {}'.format(linear_regression(x_malig_d, y_malig_d)[1]))
        print('\tBenig R2 = {}'.format(linear_regression(x_benig_d, y_benig_d)[1]))

        print('\tCross R = {}'.format(pearsonr(x_cross_d, y_cross_d)[0]))
        print('\tMalig R = {}'.format(pearsonr(x_malig_d, y_malig_d)[0]))
        print('\tBenig R = {}'.format(pearsonr(x_benig_d, y_benig_d)[0]))


        # Plot outputs
        plt.figure()
        if sub: plt.subplot(311)
        plt.scatter(x_cross_d, y_cross_d, color='blue',  alpha=0.5, s=10)
        if sub: plt.subplot(312)
        plt.scatter(x_malig_d, y_malig_d, color='red',   alpha=0.2, s=10)
        plt.ylabel(Y)
        if sub: plt.subplot(313)
        plt.scatter(x_benig_d, y_benig_d, color='green', alpha=0.1, s=10)
        plt.xlabel(X)

        #if X is 'embed' and Y is 'malig':
        #    plt.figure()
        #    plt.plot(np.histogram2d(np.squeeze(x_malig_d), np.squeeze(y_malig_d), [5, 8])[0].T)


        if sub:
            plt.figure()
            plt.title('Cross')
            plt.plot(np.histogram2d(np.squeeze(x_cross_d), np.squeeze(y_cross_d), [5,5], normed=True)[0].T)
            plt.xlabel('Malig Rating')
            plt.legend(['0', '1', '2', '3', '4'])

            plt.figure()
            plt.title('Malig')
            plt.plot(np.histogram2d(np.squeeze(x_malig_d), np.squeeze(y_malig_d), [5,5], normed=True)[0].T)
            plt.xlabel('Malig Rating')
            plt.legend(['0', '1', '2', '3', '4'])

            plt.figure()
            plt.title('Benig')
            plt.plot(np.histogram2d(np.squeeze(x_benig_d), np.squeeze(y_benig_d), [5,5], normed=True)[0].T)
            plt.xlabel('Malig Rating')
            plt.legend(['0', '1', '2', '3', '4'])

    def scatter_old(self):

        maskMalig = self.labels == 1
        maskBenig = self.labels == 0

        embed_method = 'euclidean'
        rating_method = 'euclidean'  # 'chebyshev'

        embed_benig_d = flatten_dm(calc_distance_matrix(self.embedding[maskBenig], embed_method))
        embed_malig_d = flatten_dm(calc_distance_matrix(self.embedding[maskMalig], embed_method))
        embed_cross_d = flatten_dm(
            calc_cross_distance_matrix(self.embedding[maskMalig], self.embedding[maskBenig], embed_method))

        rating = np.array(self.rating)
        rating_benig_d = flatten_dm(calc_distance_matrix(rating[maskBenig], rating_method))
        rating_malig_d = flatten_dm(calc_distance_matrix(rating[maskMalig], rating_method))
        rating_cross_d = flatten_dm(calc_cross_distance_matrix(rating[maskMalig], rating[maskBenig], rating_method))

        # Plot outputs
        plt.figure()
        plt.scatter(embed_cross_d, rating_cross_d, color='blue')
        plt.scatter(embed_malig_d, rating_malig_d, color='red')
        plt.scatter(embed_benig_d, rating_benig_d, color='green')
        plt.xlabel('Embed')
        plt.ylabel('Rating')


    def linear_regression_by_maligancy(self, embed_method='euclidean', rating_method='chebyshev'):
        maskMalig = self.labels == 1
        maskBenig = self.labels == 0

        embed_benig_d = flatten_dm(calc_distance_matrix(self.embedding[maskBenig], embed_method))
        embed_malig_d = flatten_dm(calc_distance_matrix(self.embedding[maskMalig], embed_method))

        rating = np.array(self.rating)
        rating_benig_d = flatten_dm(calc_distance_matrix(rating[maskBenig], rating_method))
        rating_malig_d = flatten_dm(calc_distance_matrix(rating[maskMalig], rating_method))
        plt.figure()
        plt.scatter(embed_malig_d, rating_malig_d, color='red')
        plt.scatter(embed_benig_d, rating_benig_d, color='green')

    def malig_regression(self, method = 'correlation'):
        #size = self.embed_distance_matrix.shape[0]
        malig_rating = [calc_rating(meta, 'malig') for meta in self.meta_data]
        malig_rating = np.array(malig_rating).reshape(-1,1)
        malig_rating_distance_matrix = calc_distance_matrix(malig_rating, method)

        malig_dist  = flatten_dm(malig_rating_distance_matrix)
        embed_dist  = flatten_dm(self.embed_distance_matrix)
        rating_dist = flatten_dm(self.rating_distance_matrix)

        plt.figure()
        plt.subplot(211)
        plt.title('embed-malig')
        plt.scatter(malig_dist, embed_dist, color='black')
        plt.subplot(212)
        plt.title('rating-malig')
        plt.scatter(malig_dist, rating_dist, color='black')

    def correlate(self, X, Y):
        x_dm, x_dist = self.load_distance_matrix(X, flat=True)
        y_dm, y_dist = self.load_distance_matrix(Y, flat=True)

        print('{}[{}]-{}[{}]:'. format(X, x_dist, Y, y_dist))

        pear, pear_p    = pearsonr(x_dm, y_dm)[0][0],   pearsonr(x_dm, y_dm)[1][0]
        spear, spear_p  = spearmanr(x_dm, y_dm)[0],     spearmanr(x_dm, y_dm)[1]
        kend, kend_p    = kendalltau(x_dm, y_dm)[0],    kendalltau(x_dm, y_dm)[1]

        print('\tPearson =\t {:.2f}, with p={:.2f}'.  format(pear,  pear_p))
        print('\tSpearman =\t {:.2f}, with p={:.2f}'. format(spear, spear_p))
        print('\tKendall =\t {:.2f}, with p={:.2f}'.  format(kend,  kend_p))

        return pear, spear, kend

    def correlate_retrieval(self, X, Y, round=False, verbose=True):
        x_dm, x_dist = self.load_distance_matrix(X, flat=False)
        y_dm, y_dist = self.load_distance_matrix(Y, flat=False)

        if verbose:
            print('{}[{}]-{}[{}]:'. format(X, x_dist, Y, y_dist))

        pear = []
        spear = []
        kend = []
        if round:
            x_dm, y_dm = np.round(x_dm), np.round(y_dm)
        for x, y in zip(x_dm, y_dm):
            pear  += [pearsonr(x, y)[0]]
            #spear += [spearmanr(x, y)[0]]
            kend  += [kendalltau(x, y)[0]]

        P = np.mean(pear), np.std(pear)
        S = None, None  # np.mean(spear), np.std(spear)
        K = np.mean(kend), np.std(kend)
        if verbose:
            print('\tPearson =\t {:.2f} ({:.2f})'.  format(P[0], P[1]))
            #print('\tSpearman =\t {:.2f} ({:.2f})'. format(S[0], S[1]))
            print('\tKendall =\t {:.2f} ({:.2f})'.  format(K[0], K[1]))

        return P, S, K

    def kendall_histogram(self, X, Y):
        x_dm, x_dist = self.load_distance_matrix(X, flat=False)
        y_dm, y_dist = self.load_distance_matrix(Y, flat=False)
        print('{}[{}]-{}[{}]:'. format(X, x_dist, Y, y_dist))

        kend = [kendalltau(x, y)[0] for x,y in zip(x_dm, y_dm)]

        hist = np.histogram(kend)
        K_Y = hist[0] / np.sum(hist[0])
        K_X = 0.5*(hist[1][:-1]+hist[1][1:])

        return K_X, K_Y

if __name__ == "__main__":
    #
    # Current Metrics:
    #   'chebyshev'
    #   'euclidean'
    #   'cosine'
    #   'corrlation'
    #
    # To evaluate similarity of two Distance-Metrices:
    #   Kendall tau distance
    #   Spearman's rank correlation
    #   Distance Correlation
    from Network import FileManager

    Embed = FileManager.Embed('siam')

    Reg = RatingCorrelator(Embed(run='064X',epoch=30,dset='Valid'))

    Reg.evaluate_embed_distance_matrix(method='euclidean')

    Reg.evaluate_rating_space()
    Reg.evaluate_rating_distance_matrix(method='euclidean')

    Reg.linear_regression()
    Reg.scatter('embed', 'rating', xMethod="euclidean", yMethod='euclidean', sub=True)
    #Reg.scatter('malig', 'rating', yMethod='euclidean', sub=True)
    #Reg.scatter('embed', 'malig', sub=True)
    #Reg.malig_regression(method='euclidean')

    Reg.correlate('malig', 'rating')
    Reg.correlate('embed', 'malig')
    Reg.correlate('embed', 'rating')

    plt.show()