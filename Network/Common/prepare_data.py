import numpy as np
try:
    from Network.dataUtils import rating_normalize, reorder
except:
    from dataUtils import rating_normalize, reorder


def prepare_data(data, rating_format='raw', reshuffle=False, verbose = 0, scaling="none"):
    # Entry:
    # 0 'patch'
    # 1 'mask'
    # 2 'class'
    # 3 'info'
    # 4 'size
    # 5 'rating'
    # 6 'rating_weights'
    # 7 'z'

    N = len(data)
    old_size = data[0][0].shape

    # ============================
    #   data: images and masks
    # ============================
    images = [np.expand_dims(entry[0], axis=-1) for entry in data]
    masks  = [np.expand_dims(entry[1], axis=-1) for entry in data]

    if verbose:
        print('prepare_data:')
        print("\tImage size changed from {} to {}".format(old_size, images[0].shape))
        print("\tImage Range = [{:.1f}, {:.1f}]".format(np.max(images[0]), np.min(images[0])))
        print("\tMasks Range = [{}, {}]".format(np.max(masks[0]), np.min(masks[0])))

    # ============================
    #   labels: classes and ratings
    # ============================

    classes = np.array([entry[2] for entry in data]).reshape(N, 1)

    rating_weights = None
    if rating_format == 'raw':
        ratings = np.array([rating_normalize(entry[5], scaling) for entry in data])
        rating_weights = np.array([entry[6] for entry in data])
    elif rating_format == 'mean':
        ratings  = np.array([rating_normalize(np.mean(entry[5], axis=0), scaling) for entry in data]).reshape(N, 9)
    elif rating_format == 'w_mean':
        w_mean = lambda R, W: np.sum(np.diag(W).dot(R) / np.sum(W), axis=0)
        ratings = np.array([rating_normalize(w_mean(entry[5], entry[6]), scaling) for entry in data]).reshape(N, 9)
    else:
        print("ERR: Illegual rating_format given ({})".format(rating_format))
        assert (False)

    if verbose:
        print("benign:{}, malignant: {}, unknown: {}".format(
                    np.count_nonzero(classes == 0),
                    np.count_nonzero(classes == 1),
                    np.count_nonzero(classes == 2)))

    # ============================
    #   meta: meta, nodule-size, slice confidence and z-value
    # ============================

    # for nodule-size use the rescaled mask area
    #
    # nodule_size = np.array([entry[4] for entry in data]).reshape(N, 1)
    # sorted_size = np.sort(nodule_size, axis=0).flatten()
    # L = len(sorted_size)
    # tresh = sorted_size[range(0, L, L//5)]
    nodule_size = np.array([np.count_nonzero(q) for q in masks]).reshape(N, 1) * 0.5 * 0.5
    tresh = [0, 15, 30, 60, 120]
    nodule_size = np.digitize(nodule_size, tresh)

    z = np.array([entry[7] for entry in data]).reshape(N, 1)

    # confidence
    # only relevant for full dataset and should first be reconsidered
    # conf = np.array([np.min(entry[6]) for entry in data])
    # mean rating based objective
    conf = 1 - .5*np.array([rating_normalize(np.std(entry[5], axis=0).mean(), scaling) for entry in data])

    meta = [entry[3] for entry in data]

    if reshuffle:
        new_order = np.random.permutation(N)
        # print('permutation: {}'.format(new_order[:20]))
        images = reorder(images, new_order)
        masks = reorder(masks, new_order)
        classes = classes[new_order]
        ratings = ratings[new_order]
        rating_weights = rating_weights[new_order] if rating_weights is not None else None
        meta = reorder(meta, new_order)
        nodule_size = nodule_size[new_order]
        z = z[new_order]
        conf = conf[new_order]

    return images, ratings, classes, masks, meta, conf, nodule_size, rating_weights, z


