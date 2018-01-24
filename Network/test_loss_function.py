import numpy as np
from keras import objectives
from keras import backend as K

from Network.siameseArch import contrastive_loss
from Network.tripletArch import triplet_loss
from Network.metrics import pearson_correlation
from scipy.stats import pearsonr

_EPSILON = K.epsilon()


def _loss_tensor(y_true, y_pred):
    return triplet_loss(y_true, y_pred)
    #return contrastive_loss(y_true, y_pred)
    #return pearson_correlation(y_true, y_pred)
    #y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    #out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    #return K.mean(out, axis=-1)

def _loss_np(y_true, y_pred):
    # triplet loss
    y2 = np.square(y_pred)
    loss = np.expand_dims(y2.dot(np.array([1, -1])), axis=-1)
    loss = np.log1p(np.exp(loss))
    return loss
    # contrastive_loss
    #margin = 1
    #return (1 - y_true) * np.square(y_pred) + y_true * np.square(np.maximum(margin - y_pred, 0))
    #return pearsonr(y_true, y_pred)
    #y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    #out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    #return np.mean(out, axis=-1)

def check_loss(_shape):
    if _shape == '2d':
        shape = (69, 2)
    elif _shape == '3d':
        shape = (5, 6, 7)
    elif _shape == '4d':
        shape = (8, 5, 6, 7)
    elif _shape == '5d':
        shape = (9, 8, 5, 6, 7)

    y_a = np.random.random(shape)
    y_b = np.random.random(shape)

    out1 = K.eval(_loss_tensor(K.variable(y_a), K.variable(y_b)))
    out2 = _loss_np(y_a, y_b)

    assert out1.shape == out2.shape
    assert out1.shape[0] == shape[0]
    print( np.linalg.norm(out1))
    print( np.linalg.norm(out2))
    print( np.linalg.norm(out1-out2))


def test_loss():
    #shape_list = ['2d', '3d', '4d', '5d']
    shape_list = ['2d']
    for _shape in shape_list:
        check_loss(_shape)
        print( '======================')

if __name__ == '__main__':
    test_loss()