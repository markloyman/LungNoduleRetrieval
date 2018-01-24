from keras import backend as K
K.set_image_data_format('channels_last')

smooth = 0.000001 # to avoid division by zero

siamese_margin = 1
siamese_rating_factor = 0.25
triplet_margin = 1.0

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

# ------------------------ #
# ==== Categorical Metrics ==== #
# ------------------------ #


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f = 2.0 * p * r / (p + r + smooth)
    return f


def sensitivity(y_true, y_pred):
    # equivalent to recall
    #   TP / (TP + FN)
    y_true_  = K.argmax(y_true, axis=1)
    y_pred_  = K.argmax(y_pred, axis=1)
    TP      = K.cast(K.sum(y_true_ * y_pred_), K.floatx())
    FN      = K.cast(K.sum(y_true_ * (1-y_pred_)), K.floatx())
    sens    = TP / (TP + FN + smooth)
    return sens


def specificity(y_true, y_pred):
    #   TN / (TN + FP)
    y_true_  = K.argmax(y_true, axis=1)
    y_pred_  = K.argmax(y_pred, axis=1)
    TN      = K.cast(K.sum((1-y_true_) * (1-y_pred_)), K.floatx())
    FP      = K.cast(K.sum((1-y_true_) * y_pred_), K.floatx())
    spec    = TN / (TN + FP + smooth)
    return spec


def precision(y_true, y_pred):
    #   TP / (TP + FP)
    y_true_  = K.argmax(y_true, axis=1)
    y_pred_  = K.argmax(y_pred, axis=1)
    TP = K.cast(K.sum(y_true_ * y_pred_), K.floatx())
    FP = K.cast(K.sum((1-y_true_) * y_pred_), K.floatx())
    prec = TP / (TP + FP + smooth)
    return prec


def recall(true, pred):
    # equivalent to recall
    return sensitivity(true, pred)


def fScore(y_true, y_pred, b=1):
    P = precision(y_true, y_pred)
    R = recall(y_true, y_pred)
    F = (b^2 + 1)* P*R / ( (b^2)*P+R)
    return F


'''
def sens_np(y_true, y_pred):
    intersection = np.sum(y_true * y_pred, axis=(1,2,3))
    fn = np.sum( y_true * (y_pred==0).astype('float32'), axis=(1,2,3))
    return np.mean( (intersection+ smooth) / (intersection+fn+smooth), axis=0)


def dice_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
'''

# ---------------------------- #
# ==== Multi-Clas Metrics ==== #
# ---------------------------- #

# use Macro avging for multi-class metrics

def root_mean_squared_error(y_true, y_pred):
    se   = K.sum(K.square(y_pred - y_true), axis=-1)
    rmse = K.sqrt(K.mean(se, axis=0))
    return rmse


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

# ---------------------------- #
# ==== Triplet Metrics ==== #
# ---------------------------- #

def rank_accuracy(_, y_pred):
    '''
        Assume: y_pred shape is (batch_size, 2)
    '''

    subtraction = K.constant([1, -1], shape=(2, 1))
    diff =  K.dot(y_pred, subtraction)
    loss = K.maximum(K.sign(-diff), K.constant(0))

    return loss


def kendall_correlation(_, y_pred):
    '''
        Assume: y_pred shape is (batch_size, 2)
    '''

    n = K.cast(K.shape(y_pred)[0], K.floatx())
    subtraction = K.constant([1, -1], shape=(2, 1))
    diff = K.dot(y_pred, subtraction)
    loss = K.sum(K.sign(-diff)) / n

    return loss
