from keras import backend as K
K.set_image_data_format('channels_last')

smooth = 0.000001  # to avoid division by zero
triplet_margin = 1.0

# ---------------------------- #
# ==== Triplet Metrics ==== #
# ---------------------------- #


def rank_accuracy(_, y_pred):
    '''
        Input:  y_pred shape is (batch_size, 2)
                [pos, neg]
        Output: shape (batch_size, 1)
                loss[i] = 1 if y_pred[i, 0] < y_pred[i, 1] else 0
    '''

    subtraction = K.constant([-1, 1], shape=(2, 1))
    diff =  K.dot(y_pred, subtraction)
    loss = K.maximum(K.sign(diff), K.constant(0))

    return loss


def kendall_correlation(_, y_pred):
    '''
        Input:  y_pred shape is (batch_size, 2)
        Output: shape (batch_size, 1)
    '''

    n = K.cast(K.shape(y_pred)[0], K.floatx())
    subtraction = K.constant([1, -1], shape=(2, 1))
    diff = K.dot(y_pred, subtraction)
    loss = K.sum(K.sign(-diff)) / n

    return loss
