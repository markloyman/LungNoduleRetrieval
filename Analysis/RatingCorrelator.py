import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import pairwise
from sklearn.neighbors import DistanceMetric

from LIDC.lidcUtils import calc_rating


# from scipy.spatial import distance as Distance



def calc_distance_matrix(X, method):
    if method in ['chebyshev', 'euclidean']:
        DM = DistanceMetric.get_metric(method).pairwise(X)
    elif method is 'cosine':
        DM = pairwise.cosine_distances(X)
    elif method in ['correlation', 'cityblock']:
        DM = squareform(pdist(X, method))
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

def flatten_dm(DM):
    return DM[np.triu_indices(DM.shape[0], 1)].reshape(-1, 1)


def linear_regression(x, y):
    regr = linear_model.LinearRegression()
    regr.fit(x,y)
    y_pred = regr.predict(x)

    return mean_squared_error(y, y_pred), r2_score(y, y_pred)



class RatingCorrelator:

    def __init__(self, filename, title='', set=''):
        self.title  = title
        self.set    = set

        self.images, self.embedding, self.meta_data, self.labels = pickle.load(open('./embed/{}'.format(filename),'br'))
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

    def evaluate_rating_space(self):
        self.rating = [calc_rating(meta) for meta in self.meta_data]
        self.rating_distance_matrix = None # reset after recalculating the ratings

    def evaluate_rating_distance_matrix(self, method='chebyshev'):
        self.rating_distance_matrix = calc_distance_matrix(self.rating, method)
        assert self.rating_distance_matrix.shape[0] == self.embed_distance_matrix.shape[0]
        self.rating_metric = method

    def evaluate_embed_distance_matrix(self, method='euclidean'):
        self.embed_distance_matrix = calc_distance_matrix(self.embedding, method)
        self.embed_metric = method

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
        if name is 'embed':
            xVec = self.embed_distance_matrix
            xMet = self.embed_metric
        elif name is 'rating':
            xVec = self.rating_distance_matrix
            xMet = self.rating_metric
        elif name is 'malig':
            malig_rating = [calc_rating(meta, 'malig') for meta in self.meta_data]
            malig_rating = np.array(malig_rating).reshape(-1, 1).astype('float64')
            xVec = calc_distance_matrix(malig_rating, method='euclidean')
            xMet = 'euclidean'
        else:
            assert False
        if flat:
            return flatten_dm(xVec), xMet
        else:
            return xVec, xMet

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

        print('\tPearson = {}, with p={}'.  format(pear,  pear_p))
        print('\tSpearman = {}, with p={}'. format(spear, spear_p))
        print('\tKendall = {}, with p={}'.  format(kend,  kend_p))

        return pear, spear, kend

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

    filename = 'embed_siam000-5_Train.p'

    Reg = RatingCorrelator(filename)

    Reg.evaluate_embed_distance_matrix(method='euclidean')

    Reg.evaluate_rating_space()
    Reg.evaluate_rating_distance_matrix(method='euclidean')

    #Reg.linear_regression()
    #Reg.scatter('embed', 'rating', yMethod='chebyshev')
    #Reg.scatter('malig', 'rating', yMethod='euclidean', sub=True)
    #Reg.scatter('embed', 'malig', sub=True)
    #Reg.malig_regression(method='euclidean')

    Reg.correlate('malig', 'rating')
    Reg.correlate('embed', 'malig')
    Reg.correlate('embed', 'rating')

    plt.show()