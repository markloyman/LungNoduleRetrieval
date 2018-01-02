import itertools
import numpy as np
import matplotlib.pyplot as plt
# from skimage.transform import resize
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise
from sklearn.neighbors import DistanceMetric

from Network import dataUtils


def calc_distance_matrix(X, method):
    if method in ['chebyshev', 'euclidean', 'l1', 'l2']:
        DM = DistanceMetric.get_metric(method).pairwise(X)
    elif method in ['cosine']:
        DM = pairwise.cosine_distances(X)
    elif method in ['correlation', 'cityblock', 'braycurtis', 'canberra', 'hamming', 'jaccard', 'kulsinski']:
        DM = squareform(pdist(X, method))
    elif method in ['minkowski3']:
        DM = squareform(pdist(X, 'minkowski', 3))
    elif method in ['dot']:
        DM = squareform(pdist(X, lambda u, v: np.dot(u, v)))
    elif method in ['emd']:
        from scipy.stats import wasserstein_distance
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


def MalignancyConfusionMatrix(pred, true):
    cm = confusion_matrix(dataUtils.uncategorize(true),
                          dataUtils.uncategorize(pred))
    #          Pred
    #   True  TN  FP
    #         FN  TP
    #plt.figure()

    TN = cm[0,0]
    TP = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]

    print("Accuracy: {}".format( (TP+TN) / (TP+TN+FN+FP) ))
    print("Sensitivity: {}".format(TP/(TP+FN)))
    print("Precision: {}".format(TP / (TP + FP)))
    print("Specificity: {}".format(TN / (TN + FP)))

    plt.matshow(cm, cmap=plt.cm.Blues)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.title('Maligancy Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True')
    plt.xlabel('Predicted')


def MalignancyBySize(pred, true, size):
    assert pred.shape[0] == true.shape[0]

    true = dataUtils.uncategorize(true)
    pred = dataUtils.uncategorize(pred)
    plt.figure()
    plt.title("Malignancy by Nodule Size")
    plt.scatter(size, true, c=(pred!=true).astype('uint'))
    plt.xlabel('Size [mm]')
    plt.ylabel('True Malignancy')


def calc_f1(precision, recall):
    assert (precision.shape == recall.shape)
    f1 = 2 * precision * recall / (precision + recall)
    assert(precision.shape == f1.shape)
    return f1


def history_summarize(history, label=''):
    if hasattr(history, 'history'):
        history = history.history

    keys = history.keys()
    print(keys)
    # dict_keys(['loss', 'val_sensitivity', 'val_categorical_accuracy', 'val_loss', 'val_specificity', 'val_precision', 'lr', 'precision', 'categorical_accuracy', 'sensitivity', 'specificity'])

    plt.figure()

    plt.subplot(311)
    # summarize history for sensitivity/recall
    if 'sensitivity'    in keys:
        measure = 'sensitivity'
    elif 'recall'        in keys:
        measure = 'recall'
    elif 'binary_recall_inv' in keys:
        measure = 'binary_recall_inv'
    else:
        measure = None

    if measure is not None:
        plt.plot(history[measure])
        plt.plot(history['val_{}'.format(measure)])
        plt.title('{}: {}'.format(label, measure))
        plt.ylabel(measure)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.grid(True)

    plt.subplot(312)
    # summarize history for precision
    if  'precision' in keys:
        measure = 'precision'
    elif 'binary_precision_inv' in keys:
        measure = 'binary_precision_inv'
    else:
        measure = None

    if measure is not None:
        plt.plot(history[measure])
        plt.plot(history['val_{}'.format(measure)])
        plt.title('{}: {}'.format(label, measure))
        plt.ylabel(measure)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.grid(True)

    plt.subplot(313)
    '''
    # summarize history for loss and lr
    if 'accuracy' in keys:
        measure = 'accuracy'
    elif 'binary_accuracy' in keys:
        measure = 'binary_accuracy'
    #if 'binary_assert' in keys:
    #    measure = 'binary_assert'
    elif  'loss' in keys:
        measure = 'loss'
    else:
        measure = None

    if measure is not None:
        plt.plot(history[measure])
        plt.plot(history['val_{}'.format(measure)])
        plt.title('{}: {}'.format(label, measure))
        plt.ylabel(measure)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.grid(True)
    '''
    if 'precision' in keys and 'recall' in keys:
        p = 'precision'
        r = 'recall'
        plt.plot(calc_f1(history[p], history[r]))
        plt.plot(calc_f1(history['val_{}'.format(p)], history['val_{}'.format(r)]))
        plt.title('{}: {}'.format(label, measure))
        plt.ylabel(measure)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.grid(True)


def calc_embedding_statistics(embedding, data_dim=0, title=''):
    max_ = np.max(embedding, axis=data_dim)
    min_ = np.min(embedding, axis=data_dim)
    mean_ = np.mean(embedding, axis=data_dim)
    std_ = np.std(embedding, axis=data_dim)
    absmean_ = np.mean(np.abs(embedding), axis=data_dim)
    range_ = max_ - min_

    print('Dim = {}'.format(max_.shape))

    plt.figure(title)
    bins_ = 20

    plt.subplot(321)
    plt.title('Range: [{:.3f},{:.1f}], Mean: {:.2f}'.format(np.min(min_), np.max(min_), np.mean(min_)))
    plt.ylabel('Min')
    plt.hist(min_, bins=bins_)

    plt.subplot(322)
    plt.title('Range: [{:.3f},{:.1f}], Mean: {:.2f}'.format(np.min(max_), np.max(max_), np.mean(max_)))
    plt.ylabel('Max')
    plt.hist(max_, bins=bins_)

    plt.subplot(323)
    plt.title('Range: [{:.3f},{:.1f}], Mean: {:.2f}'.format(np.min(mean_), np.max(mean_), np.mean(mean_)))
    plt.ylabel('Mean')
    plt.hist(mean_, bins=bins_)

    plt.subplot(324)
    plt.title('Range: [{:.3f},{:.1f}], Mean: {:.2f}'.format(np.min(std_), np.max(std_), np.mean(std_)))
    plt.ylabel('std')
    plt.hist(std_, bins=bins_)

    plt.subplot(325)
    plt.title('Range: [{:.1f},{:.1f}], Mean: {:.1f}'.format(np.min(absmean_), np.max(absmean_), np.mean(absmean_)))
    plt.ylabel('AbsMean')
    plt.hist(absmean_, bins=bins_)

    plt.subplot(326)
    plt.title('Range: [{:.1f},{:.1f}], Mean: {:.1f}'.format(np.min(range_), np.max(range_), np.mean(range_)))
    plt.ylabel('Range')
    plt.hist(range_, bins=bins_)
