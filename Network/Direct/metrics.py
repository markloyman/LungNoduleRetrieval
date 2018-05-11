from keras import backend as K
K.set_image_data_format('channels_last')


# ------------------------ #
# ==== Categorical Metrics ==== #
# ------------------------ #

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f = 2.0 * p * r / (p + r + K.epsilon())
    return f


def sensitivity(y_true, y_pred):
    # equivalent to recall
    #   TP / (TP + FN)
    y_true_  = K.argmax(y_true, axis=1)
    y_pred_  = K.argmax(y_pred, axis=1)
    TP      = K.cast(K.sum(y_true_ * y_pred_), K.floatx())
    FN      = K.cast(K.sum(y_true_ * (1-y_pred_)), K.floatx())
    sens    = TP / (TP + FN + K.epsilon())
    return sens


def specificity(y_true, y_pred):
    #   TN / (TN + FP)
    y_true_  = K.argmax(y_true, axis=1)
    y_pred_  = K.argmax(y_pred, axis=1)
    TN      = K.cast(K.sum((1-y_true_) * (1-y_pred_)), K.floatx())
    FP      = K.cast(K.sum((1-y_true_) * y_pred_), K.floatx())
    spec    = TN / (TN + FP + K.epsilon())
    return spec


def precision(y_true, y_pred):
    #   TP / (TP + FP)
    y_true_  = K.argmax(y_true, axis=1)
    y_pred_  = K.argmax(y_pred, axis=1)
    TP = K.cast(K.sum(y_true_ * y_pred_), K.floatx())
    FP = K.cast(K.sum((1-y_true_) * y_pred_), K.floatx())
    prec = TP / (TP + FP + K.epsilon())
    return prec


def recall(true, pred):
    # equivalent to recall
    return sensitivity(true, pred)


def fScore(y_true, y_pred, b=1):
    P = precision(y_true, y_pred)
    R = recall(y_true, y_pred)
    F = (b^2 + 1)* P*R / ( (b^2)*P+R)
    return F


# ---------------------------- #
# ==== Multi-Clas Metrics ==== #
# ---------------------------- #

# use Macro avging for multi-class metrics

def root_mean_squared_error(y_true, y_pred):
    se   = K.sum(K.square(y_pred - y_true), axis=-1)
    rmse = K.sqrt(K.mean(se, axis=0))
    return rmse


def multitask_accuracy(y_true, y_pred):
    y_pred  = K.round(y_pred)
    eq      = K.equal(y_pred, y_true)
    eq      = K.all(eq, axis=-1)
    acc     = K.sum(K.cast(eq, K.floatx()))
    return acc
