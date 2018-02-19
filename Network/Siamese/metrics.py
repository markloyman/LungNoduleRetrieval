from keras import backend as K
K.set_image_data_format('channels_last')

smooth = 0.000001 # to avoid division by zero

siamese_margin = 1
siamese_rating_factor = 0.25

# ------------------------ #
# ==== Binary Metrics ==== #
# ------------------------ #


def binary_assert(y_true, y_pred, margin=siamese_margin):
    y_pred_ = K.round(K.clip(y_pred/margin, 0, 1))
    #ones   = K.cast(K.equal(y_pred, 1), K.floatx())
    #zeros  = K.cast(K.equal(y_pred, 0), K.floatx())
    #y_pred =  K.sum(ones) + K.sum(zeros)
    pred   = K.cast(K.equal(y_pred_, 1), K.floatx())
    true   = K.cast(K.equal(y_true, 1), K.floatx())
    y_pred_ =  K.sum(pred*true)/K.sum(true)
    return y_pred_


def binary_f1(y_true, y_pred, margin=siamese_margin):
    p = binary_precision(y_true, y_pred, margin=margin)
    r = binary_recall(y_true, y_pred)
    f1 = 2.0 * p * r / (p + r)
    return f1


def binary_f1_inv(y_true, y_pred, margin=siamese_margin):
    p = binary_precision_inv(y_true, y_pred, margin=margin)
    r = binary_recall_inv(y_true, y_pred)
    f1 = 2.0 * p * r / (p + r)
    return f1


def binary_accuracy(y_true, y_pred, margin=siamese_margin):
    y_pred_ = K.round(K.clip(y_pred/margin, 0, 1))
    return K.mean(K.equal(y_true, y_pred_), axis=-1)


def binary_diff(y_true, y_pred, margin=siamese_margin):
    y_pred_ = K.round(K.clip(y_pred/margin, 0, 1))
    return K.mean(K.equal(y_true, y_pred_), axis=-1)


def binary_sensitivity(y_true, y_pred, margin=siamese_margin):
    # equivalent to recall
    #   TP / (TP + FN)
    #y_true  = K.argmax(y_true, axis=1)
    #y_pred  = K.argmax(y_pred, axis=1)
    y_pred_  = K.round(K.clip(y_pred/margin, 0, 1))
    TP      = K.cast(K.sum(y_true * y_pred_), K.floatx())
    FN      = K.cast(K.sum(y_true * (1-y_pred_)), K.floatx())
    sens    = TP / (TP + FN + smooth)
    return sens


def binary_sensitivity_inv(y_true, y_pred, margin=siamese_margin):
    y_pred_  = K.round(K.clip(y_pred/margin, 0, 1))
    TP      = K.cast(K.sum((1-y_true) * (1-y_pred_)), K.floatx())
    FN      = K.cast(K.sum((1-y_true) * y_pred_), K.floatx())
    sens    = TP / (TP + FN + smooth)
    return sens


def binary_recall(true, pred):
    # equivalent to recall
    return binary_sensitivity(true, pred)

def binary_recall_inv(true, pred):
    # equivalent to recall
    return binary_sensitivity_inv(true, pred)


def binary_precision(y_true, y_pred, margin = siamese_margin):
    #   TP / (TP + FP)
    y_pred_ = K.round(K.clip(y_pred / margin, 0, 1))
    TP = K.cast(K.sum(y_true * y_pred_), K.floatx())
    FP = K.cast(K.sum((1-y_true) * y_pred_), K.floatx())
    prec = TP / (TP + FP + smooth)
    return prec


def binary_precision_inv(y_true, y_pred, margin=siamese_margin):
    #   TP / (TP + FP)
    y_pred_ = K.round(K.clip(y_pred / margin, 0, 1))
    TP = K.cast(K.sum((1-y_true) * (1-y_pred_)), K.floatx())
    FP = K.cast(K.sum(y_true * (1-y_pred_)), K.floatx())
    prec = TP / (TP + FP + smooth)
    return prec


# Distance Metric

def pearson_correlation(y_true, y_pred):
    y_true = K.cast(y_true, K.floatx())
    sum_true = K.sum(y_true, axis=0)
    sum2_true = K.sum(K.square(y_true), axis=0)

    y_pred = K.cast(y_pred, K.floatx())
    sum_pred = K.sum(y_pred, axis=0)
    sum2_pred = K.sum(K.square(y_pred), axis=0)

    prod      = K.sum(y_true*y_pred, axis=0)
    #print(K.is_keras_tensor(sum_true))
    #print(K.is_keras_tensor(sum2_true))
    n = K.cast(K.shape(y_true)[0], K.floatx())
    #n= 64
    eps = 0.0000001
    corr =  (n*prod - sum_true*sum_pred)
    corr /= K.sqrt(n * sum2_true - sum_true * sum_true + eps)
    corr /= K.sqrt(n * sum2_pred - sum_pred * sum_pred + eps)

    return corr