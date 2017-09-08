from keras import backend as K
K.set_image_data_format('channels_last')

smooth = 0.000001 # to avoid division by zero



def binary_accuracy(y_true, y_pred, margin=5):
    y_pred = K.clip(y_pred/margin, 0, 1)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

# ------------------------ #
# ==== Binary Metrics ==== #
# ------------------------ #

def sensitivity(y_true, y_pred):
# equivalent to recall
#   TP / (TP + FN)
    y_true  = K.argmax(y_true, axis=1)
    y_pred  = K.argmax(y_pred, axis=1)
    TP      = K.cast(K.sum(y_true * y_pred), K.floatx())
    FN      = K.cast(K.sum(y_true * (1-y_pred)), K.floatx())
    sens    = TP / (TP + FN + smooth)
    return sens


def specificity(y_true, y_pred):
#   TN / (TN + FP)
    y_true  = K.argmax(y_true, axis=1)
    y_pred  = K.argmax(y_pred, axis=1)
    TN      = K.cast(K.sum((1-y_true) * (1-y_pred)), K.floatx())
    FP      = K.cast(K.sum((1-y_true) * y_pred), K.floatx())
    spec    = TN / (TN + FP + smooth)
    return spec


def precision(y_true, y_pred):
#   TP / (TP + FP)
    y_true  = K.argmax(y_true, axis=1)
    y_pred  = K.argmax(y_pred, axis=1)
    TP = K.cast(K.sum(y_true * y_pred), K.floatx())
    FP = K.cast(K.sum((1-y_true) * y_pred), K.floatx())
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

