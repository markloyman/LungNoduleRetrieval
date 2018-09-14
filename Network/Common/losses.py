from keras import backend as K
from keras.callbacks import Callback
import tensorflow as tf
from functools import partial

# covariance loss
def dispersion_loss(_, y_pred):
    y_pred = y_pred - K.mean(y_pred, axis=0, keepdims=True) # + K.random_uniform(shape=y_pred.shape[1:], minval=1e-12, maxval=1e-6)
    cov = K.dot(K.transpose(y_pred), y_pred)

    #cov_ = (cov + tf.transpose(cov, perm=[0, 2, 1])) / 2.
    #cov_ = (cov + tf.transpose(cov)) / 2.
    #e, v = tf.self_adjoint_eig(cov_)
    #e = tf.where(e > 1e-14, e, 1e-14 * tf.ones_like(e))
    #cov_ = tf.matmul(tf.matmul(v, tf.matrix_diag(e), transpose_a=True), v)

    #cov_ = 0.5 * (cov + tf.transpose(cov) - tf.diag(tf.diag_part(cov)))

    dispersion = -1 * tf.linalg.det(cov)

    return dispersion


def correlation_features_loss_adapter(batch_size=32):
    func = partial(correlation_features_loss, batch_size=batch_size)
    func.__name__ = 'features_corr'
    return func


def correlation_features_loss(_, y_pred, batch_size=32):
    # correlation matrix
    y_pred = y_pred - K.mean(y_pred, axis=0, keepdims=True)
    y_pred = y_pred / (K.std(y_pred, axis=0, keepdims=True) + 1e-6)
    corr_matrix = K.dot(K.transpose(y_pred), y_pred)
    # off-diagonal mask
    N, F = batch_size, K.int_shape(y_pred)[1]
    mask = K.ones_like(corr_matrix) - K.eye(size=F)
    # l1
    elem = K.flatten(corr_matrix * mask)
    reg = K.mean(K.abs(elem)) / K.constant(N)
    return reg


def correlation_samples_loss_adapter(batch_size=32):
    func = partial(correlation_samples_loss, batch_size=batch_size)
    func.__name__ = 'sample_corr'
    return func


def correlation_samples_loss(_, y_pred, batch_size=32):
    # correlation matrix
    y_pred = y_pred - K.mean(y_pred, axis=1, keepdims=True)
    y_pred = y_pred / (K.std(y_pred, axis=1, keepdims=True) + 1e-6)
    corr_matrix = K.dot(y_pred, K.transpose(y_pred))
    # off-diagonal mask
    N, F = batch_size, K.int_shape(y_pred)[1]
    mask = K.ones_like(corr_matrix) - K.eye(size=N)
    # l1
    elem = K.flatten(corr_matrix * mask)
    reg = K.mean(K.abs(elem)) / K.constant(F)
    return reg


def stdev_loss(_, y_pred):
    s = K.std(y_pred, axis=0)
    return K.mean(s)


def l2DM(x):
    r = K.sum(x*x, 1)
    r = K.reshape(r, [-1, 1])  # turn r into column vector
    dm = r - 2*K.dot(x, K.transpose(x)) + K.transpose(r)
    #dm = K.sqrt(dm)
    return dm

def pearson_correlation(y_true, y_pred):
    # distance matrix based on cosine-distance (assuming embeddings are all normalized)
    #dm_true = K.dot(y_true, K.transpose(y_true))
    #dm_pred = K.dot(y_pred, K.transpose(y_pred))
    # l2 distance matrix
    #dm_true = l2DM(y_true)
    dm_true = y_true
    dm_pred = l2DM(y_pred)

    #y_true = K.cast(y_true, K.floatx())
    sum_true = K.sum(dm_true, axis=1)
    sum2_true = K.sum(K.square(dm_true), axis=1)

    #y_pred = K.cast(y_pred, K.floatx())
    sum_pred = K.sum(dm_pred, axis=1)
    sum2_pred = K.sum(K.square(dm_pred), axis=1)

    prod = K.sum(dm_true*dm_pred, axis=1)
    n = K.cast(K.shape(y_true)[0], K.floatx())

    corr =  (n*prod - sum_true*sum_pred)
    corr /= K.sqrt(n * sum2_true - sum_true * sum_true + K.epsilon())
    corr /= K.sqrt(n * sum2_pred - sum_pred * sum_pred + K.epsilon())

    return -corr


class LossWeightSchedualer(Callback):
    def __init__(self, weight_list, schedule):
        self.weight_list = weight_list
        self.schedule = schedule

    def on_epoch_end(self, epoch, logs={}):
        for rule in self.schedule:
            assert type(rule) is dict
            if rule['epoch'] == epoch:
                for i, weight in enumerate(rule['weights']):
                    K.set_value(self.weight_list[i], weight)
                print("Epoch {}, loss weights update: {}".format(epoch, rule['weights']))
