import numpy as np
from scipy.spatial.distance import cdist
from functools import reduce
try:
    from Network.Common import prepare_data
    from Network.dataUtils import rating_clusters_distance, rating_clusters_distance_and_std, reorder
except:
    from Common import prepare_data
    from dataUtils import rating_clusters_distance, rating_clusters_distance_and_std, reorder


def select_different_pair(class_A, class_B, n):
    if type(class_A[0]) is np.ndarray:
        concat = lambda x: np.concatenate(x)
    else:
        concat = lambda x: reduce(lambda r, s: r + s, x)

    different = [(a, b) for a, b in zip(
        concat( [[a, a] for a in class_A[:n]] )[:-1],
        concat( [[b, b] for b in class_B[:n]] )[1:])
                 ] + [(class_A[:n][-1], class_B[0])]
    assert(2*n == len(different))
    return different, len(different)


def select_pairs(class_A):
    pairs = [(a1, a2) for a1, a2 in zip(class_A[:-1], class_A[1:])] + [(class_A[-1], class_A[0])]
    return pairs


def select_same_pairs(class_A, class_B):
    same = [(a1, a2) for a1, a2 in zip(class_A[:-1], class_A[1:])] + [(class_A[-1], class_A[0])]
    sa_size = len(same)
    same += [(b1, b2) for b1, b2 in zip(class_B[:-1], class_B[1:])] + [(class_B[-1], class_B[0])]
    sb_size = len(same) - sa_size
    assert(len(same) == (len(class_A)+len(class_B)))
    return same, sa_size, sb_size


def prepare_data_siamese(data, objective="malignancy", rating_distance='mean', balanced=False, verbose= 0):
    if verbose:
        print('prepare_data_siamese:')
    images, ratings, classes, masks, meta, conf, nod_size, _, _ = \
        prepare_data(data, rating_format='raw', reshuffle=True, verbose=verbose)

    N = len(images)
    benign_filter = np.where(classes == 0)[0]
    malign_filter = np.where(classes == 1)[0]
    M = min(benign_filter.shape[0], malign_filter.shape[0])

    if balanced:
        malign_filter = malign_filter[:M]
        benign_filter = benign_filter[:M]

    #   Handle Patches
    # =========================

    imgs_benign, imgs_malign = [images[x] for x in benign_filter], [images[x] for x in malign_filter]
    different, d_size = select_different_pair(imgs_benign, imgs_malign, n=M)
    same, sb_size, sm_size = select_same_pairs(imgs_benign, imgs_malign)

    image_pairs = same + different
    image_sub1 = [pair[0] for pair in image_pairs]
    image_sub2 = [pair[1] for pair in image_pairs]

    if objective == "malignancy":
        similarity_labels = np.concatenate([np.repeat(0, len(same)),
                                        np.repeat(1, len(different))])
    elif objective == "rating":
        lbls_benign, lbls_malign = ratings[benign_filter], ratings[malign_filter]
        diff_lbls, d_size = select_different_pair(lbls_benign, lbls_malign, n=M)
        same_lbls, sb_size, sm_size = select_same_pairs(lbls_benign, lbls_malign)

        label_pairs = same_lbls + diff_lbls
        if rating_distance == 'mean':
            similarity_labels = np.array([np.sqrt((a-b).dot(a-b)) for a, b in label_pairs])
        elif rating_distance == 'clusters':
            assert False
        else:
            assert False
    else:
        print("ERR: {} is not a valid objective".format(objective))
        assert(False)

    #   Handle Masks
    # =========================

    mask_benign, mask_malign = [masks[x] for x in benign_filter], [masks[x] for x in malign_filter]
    different_mask, d = select_different_pair(mask_benign, mask_malign, n=M)
    same_mask, sb, sm = select_same_pairs(mask_benign, mask_malign)
    assert(d == d_size)
    assert ( (sb==sb_size) and (sm==sm_size) )

    mask_pairs = same_mask + different_mask
    mask_sub1 = [pair[0] for pair in mask_pairs]
    mask_sub2 = [pair[1] for pair in mask_pairs]

    #   Handle Meta
    # =========================
    meta_benign, meta_malign = reorder(meta, benign_filter), reorder(meta, malign_filter)
    different_meta, d = select_different_pair(meta_benign, meta_malign, n=M)
    same_meta, sb, sm = select_same_pairs(meta_benign, meta_malign)
    assert (d == d_size)
    assert ((sb == sb_size) and (sm == sm_size))

    meta_pairs = same_meta + different_meta
    meta_sub1, meta_sub2 = zip(*meta_pairs)

    #   Final touch
    # =========================

    size = similarity_labels.shape[0]
    assert size == len(image_sub1)
    assert size == len(image_sub2)

    # assign confidence classes (weights are resolved online per batch)
    confidence = np.concatenate([  np.repeat('SB', sb_size),
                                   np.repeat('SM', sm_size),
                                   np.repeat('D',  d_size)
                                ])

    if verbose:
        print("{} pairs of same / {} pairs of different. {} total number of pairs".format(len(same), len(different), size))

    new_order = np.random.permutation(size)

    return (    (reorder(image_sub1, new_order), reorder(image_sub2, new_order)),
                similarity_labels[new_order],
                (reorder(mask_sub1, new_order), reorder(mask_sub2, new_order)),
                confidence[new_order],
                (reorder(meta_sub1, new_order), reorder(meta_sub2, new_order))
            )


def prepare_data_siamese_simple(data, siamese_rating_factor, objective="malignancy", rating_distance='mean', verbose=0):
    if verbose:
        print('prepare_data_siamese_simple:')
    images, ratings, classes, masks, meta, conf, nod_size, rating_weights, z = \
        prepare_data(data, rating_format='raw', scaling="none", reshuffle=True, verbose=verbose)

    N = len(images)

    #   Handle Patches
    # =========================

    image_pairs = select_pairs(images)
    image_sub1 = [pair[0] for pair in image_pairs]
    image_sub2 = [pair[1] for pair in image_pairs]

    #   Handle Labels
    # =========================

    rating_pairs = select_pairs(ratings)
    rating_weight_pairs = select_pairs(rating_weights)

    confidence = np.ones(len(image_pairs))

    if objective in ["rating", "rating_size"]:

        if rating_distance == 'mean':
            similarity_ratings = np.array([np.sqrt((a - b).dot(a - b)) for a, b in rating_pairs])
        elif rating_distance == 'clusters':
            similarity_ratings = []
            confidence = []
            for r1, r2 in rating_pairs:
                distance, std = rating_clusters_distance_and_std(r1, r2)
                similarity_ratings += [distance]
                confidence += [std]
            similarity_ratings = np.array(similarity_ratings)
            confidence = np.array(confidence)
            confidence = 1 - .5 * confidence / (confidence + .5)
        elif 'weighted_clusters':
            similarity_ratings = []
            confidence = []
            for r, w in zip(rating_pairs, rating_weight_pairs):
                distance, std = rating_clusters_distance_and_std(r[0], r[1], 'euclidean', weight_a=w[0], weight_b=w[1])
                similarity_ratings += [distance]
                confidence += [std]
            similarity_ratings = np.array(similarity_ratings)
            confidence = np.array(confidence)
            confidence = 1 - .5 * confidence / (confidence + .5)
        else:
            assert False
        similarity_ratings *= siamese_rating_factor

    if objective in ['size', 'rating_size']:
        size_pairs = select_pairs(nod_size)
        similarity_size = np.array([np.sqrt((a - b).dot(a - b)) for a, b in size_pairs])

        if similarity_size.ndim == 1:
            similarity_ratings = np.expand_dims(similarity_size, axis=1)

    if similarity_ratings.ndim == 1:
        similarity_ratings = np.expand_dims(similarity_ratings, axis=1)

    if objective == "rating":
        similarity_labels = similarity_ratings,
    elif objective == 'size':
        similarity_labels = similarity_size,
    elif objective == 'rating_size':
        similarity_labels = similarity_ratings, similarity_size
    else:
        print("ERR: {} is not a valid objective".format(objective))
        assert False

    #   Handle Masks
    # =========================

    mask_pairs = select_pairs(masks)
    mask_sub1 = [pair[0] for pair in mask_pairs]
    mask_sub2 = [pair[1] for pair in mask_pairs]

    #   Handle Meta
    # =========================
    meta_pairs = select_pairs(meta)
    meta_sub1 = [pair[0] for pair in meta_pairs]
    meta_sub2 = [pair[1] for pair in meta_pairs]

    #   Final touch
    # =========================

    size = similarity_labels[0].shape[0]
    assert size == len(image_sub1)
    assert size == len(image_sub2)

    # assign confidence classes (weights are resolved online per batch)
    #confidence = np.concatenate([  np.repeat('SB', sb_size),
    #                               np.repeat('SM', sm_size),
    #                               np.repeat('D',  d_size)
    #                            ])

    #confidence = np.repeat('SB', N)
    #onfidence = []
    #for r1, r2 in rating_pairs:
    #    dm = cdist(r1, r2, 'euclidean')
    #    d0 = np.max(dm, axis=0)
    #    d1 = np.max(dm, axis=1)
    #    distance = 0.5 * np.mean(d0) + 0.5 * np.mean(d1)
    #    confidence += [distance]
    #confidence = 1.0 - np.array(confidence)/(8.0 + 0.25*np.array(confidence))

    new_order = np.random.permutation(size)

    return (    (reorder(image_sub1, new_order), reorder(image_sub2, new_order)),
                tuple([s[new_order] for s in similarity_labels]),
                (reorder(mask_sub1, new_order), reorder(mask_sub2, new_order)),
                confidence[new_order],
                (reorder(meta_sub1, new_order), reorder(meta_sub2, new_order))
            )
