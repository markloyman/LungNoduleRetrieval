import numpy as np
try:
    from Network.Common import prepare_data
except:
    from Common import prepare_data


def select_balanced(some_set, labels, N, permutation):
    b = some_set[0 == labels][:N]
    m = some_set[1 == labels][:N]
    merged = np.concatenate([b, m], axis=0)
    reshuff = merged[permutation]
    return reshuff


def prepare_data_direct(data, objective='malignancy', rating_format='w_mean', rating_scale='none', weighted_rating=False, num_of_classes=2, balanced=False, verbose=0, reshuffle=True):
    rating_format = 'raw' if objective == 'distance-matrix' else rating_format
    images, ratings, classes, masks, meta, conf, nod_size, rating_weights, z = \
        prepare_data(data, rating_format=rating_format, scaling=rating_scale, verbose=verbose, reshuffle=reshuffle)

    if objective == 'malignancy':
        from keras.utils.np_utils import to_categorical
        labels = to_categorical(classes, num_of_classes)
    elif objective == 'rating':
        labels = ratings
    elif objective == 'size':
        labels = nod_size
    elif objective == 'rating_size':
        labels = ratings, nod_size
    elif objective == 'distance-matrix':
        labels = (ratings, rating_weights) if weighted_rating else ratings
    else:
        assert False

    Nb = np.count_nonzero(0 == classes)
    Nm = np.count_nonzero(1 == classes)
    #Nu = np.count_nonzero(2 == classes)
    N = np.minimum(Nb, Nm)

    if balanced:
        new_order = np.random.permutation(2 * N)
        labels_ = np.argmax(classes, axis=1)
        images = select_balanced(images, labels_, N, new_order)
        labels = select_balanced(labels, labels_, N, new_order)
        classes = select_balanced(classes, labels_, N, new_order)
        masks = select_balanced(masks, labels_, N, new_order)
        if verbose:
            Nb = np.count_nonzero(1 - np.argmax(classes, axis=1))
            Nm = np.count_nonzero(np.argmax(classes, axis=1))
            print("Balanced - Benign: {}, Malignant: {}".format(Nb, Nm))

    return images, labels, classes, masks, meta, conf