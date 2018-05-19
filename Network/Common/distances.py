import keras.backend as K


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def l1_distance(vects):
    x, y = vects
    return K.sum(K.abs(x - y), axis=1, keepdims=True)


def cosine_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return 1.0 - K.batch_dot(x, y, axes=-1)


def distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)