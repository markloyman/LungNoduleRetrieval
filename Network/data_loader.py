import pickle
import numpy as np
from functools import reduce
from scipy.misc import imresize
try:
    from Network.dataUtils import rating_normalize, crop_center
except:
    from dataUtils import rating_normalize, crop_center


# =========================
#   Load
# =========================


def load_nodule_dataset(size=128, res=1.0, apply_mask_to_patch=False, sample='Normal', configuration=None, full=False, include_unknown=False, n_groups=5):
    if configuration is None:
        return load_nodule_dataset_old_style(size=size, res=res, apply_mask_to_patch=apply_mask_to_patch, sample=sample)

    dataset_type = "Full" if full else ("Primary" if include_unknown else "Clean")
    test_id  = configuration
    valid_id = (configuration + 1) % n_groups
    testData, validData, trainData = [], [], []

    for c in range(n_groups):
        filename = '/Dataset/Dataset{}CV{}_{:.0f}-{}-{}.p'.format(dataset_type, c, size, res, sample)
        try:
            data_group = pickle.load(open(filename, 'br'))
        except:
            data_group = pickle.load(open('.'+filename, 'br'))

        #if not include_unknown:
        #    data_group = list(filter(lambda x: x['label'] < 2, data_group))

        if c == test_id:
            set = "Test"
            testData += data_group
        elif c == valid_id:
            set = "Valid"
            validData += data_group
        else:
            set = "Train"
            trainData += data_group
        print("Loaded {} entries from {} to {} set".format(len(data_group), filename, set))

    if apply_mask_to_patch:
        print('WRN: apply_mask_to_patch is for debug only')
        testData = [(entry['patch'] * (0.3 + 0.7 * entry['mask']), entry['mask'], entry['label'], entry['info'],
                     entry['size'], entry['rating']) for entry in testData]
        validData = [(entry['patch'] * (0.3 + 0.7 * entry['mask']), entry['mask'], entry['label'], entry['info'],
                      entry['size'], entry['rating']) for entry in validData]
        trainData = [(entry['patch'] * (0.3 + 0.7 * entry['mask']), entry['mask'], entry['label'], entry['info'],
                      entry['size'], entry['rating']) for entry in trainData]
    else:
        testData = [(entry['patch'], entry['mask'], entry['label'], entry['info'], entry['size'], entry['rating'])
                    for entry in testData]
        validData = [(entry['patch'], entry['mask'], entry['label'], entry['info'], entry['size'], entry['rating'])
                     for entry in validData]
        trainData = [(entry['patch'], entry['mask'], entry['label'], entry['info'], entry['size'], entry['rating'])
                     for entry in trainData]

    image_ = np.concatenate([e[0] for e in trainData])
    print("\tImage Range = [{:.1f}, {:.1f}]".format(np.max(image_), np.min(image_)))

    return testData, validData, trainData


def load_nodule_dataset_old_style(size=128, res=1.0, apply_mask_to_patch=False, sample='Normal'):

    if type(res) == str:
        filename = '/Dataset/Dataset{:.0f}-{}-{}.p'.format(size, res, sample)
    else:
        filename = '/Dataset/Dataset{:.0f}-{:.1f}-{}.p'.format(size, res, sample)

    try:
        testData, validData, trainData = pickle.load(open(filename, 'br'))
    except:
        testData, validData, trainData = pickle.load(open('.'+filename, 'br'))

    print('Loaded: {}'.format(filename))
    image_ = np.concatenate([e['patch'] for e in trainData])
    print("\tImage Range = [{:.1f}, {:.1f}]".format(np.max(image_), np.min(image_)))
    print("\tMasks Range = [{}, {}]".format(np.max(trainData[0]['mask']), np.min(trainData[0]['mask'])))
    #print("\tLabels Range = [{}, {}]".format(np.max(trainData[0]['label']), np.min(trainData[0]['label'])))

    if apply_mask_to_patch:
        print('WRN: apply_mask_to_patch is for debug only')
        testData  = [(entry['patch']*(0.3+0.7*entry['mask']), entry['mask'], entry['label'], entry['info'], entry['size'], entry['rating']) for entry in testData]
        validData = [(entry['patch']*(0.3+0.7*entry['mask']), entry['mask'], entry['label'], entry['info'], entry['size'], entry['rating']) for entry in validData]
        trainData = [(entry['patch']*(0.3+0.7*entry['mask']), entry['mask'], entry['label'], entry['info'], entry['size'], entry['rating']) for entry in trainData]
    else:
        testData  = [ (entry['patch'], entry['mask'], entry['label'], entry['info'], entry['size'], entry['rating']) for entry in testData ]
        validData = [ (entry['patch'], entry['mask'], entry['label'], entry['info'], entry['size'], entry['rating']) for entry in validData]
        trainData = [ (entry['patch'], entry['mask'], entry['label'], entry['info'], entry['size'], entry['rating']) for entry in trainData]

    return testData, validData, trainData


def load_nodule_raw_dataset(size=128, res='Legacy', sample='Normal'):
    if type(res) == str:
        filename = '/Dataset/Dataset{:.0f}-{}-{}.p'.format(size, res, sample)
    else:
        filename = '/Dataset/Dataset{:.0f}-{:.1f}-{}.p'.format(size, res, sample)

    try:
        testData, validData, trainData = pickle.load(open(filename, 'br'))
    except:
        testData, validData, trainData = pickle.load(open('.'+filename, 'br'))

    return testData, validData, trainData


# =========================
#   Prepare Data
# =========================


def reorder(a_list, order):
    return [a_list[order[i]] for i in range(len(order))]


def prepare_data(data, objective='malignancy', new_size=None, do_augment=False, categorize=0, return_meta=False, rating_confidence=False, reshuffle=False, verbose = 0, scaling="none"):
    from keras.utils.np_utils import to_categorical
    # Entry:
    # 0 'patch'
    # 1 'mask'
    # 2 'label'
    # 3 'info'

    N = len(data)
    old_size = data[0][0].shape
    if new_size is not None:
        assert(False) # imresize changes values. should be replaced
        images = np.array([imresize(entry[0], size=(new_size, new_size), interp='nearest')
                           for entry in data]).reshape(N, new_size, new_size, 1)
        masks = np.array([imresize(entry[1], size=(new_size, new_size), interp='nearest').astype('uint16')
                          for entry in data]).reshape(N, new_size, new_size, 1)
    else:
        images = np.array([entry[0] for entry in data]).reshape(N, old_size[0], old_size[1], 1)
        masks  = np.array([entry[1] for entry in data]).reshape(N, old_size[0], old_size[1], 1)
    if verbose:
        print('prepare_data:')
        print("\tImage size changed from {} to {}".format(old_size, images[0].shape))
        print("\tImage Range = [{:.1f}, {:.1f}]".format(np.max(images[0]), np.min(images[0])))
        print("\tMasks Range = [{}, {}]".format(np.max(masks[0]), np.min(masks[0])))

    if objective == 'malignancy':
        labels = np.array([entry[2] for entry in data]).reshape(N, 1)
        classes = np.copy(labels)
    elif objective == 'rating':
        labels  = np.array([rating_normalize(np.mean(entry[5], axis=0), scaling) for entry in data]).reshape(N, 9)
        classes = np.array([entry[2] for entry in data]).reshape(N, 1)
    else:
        print("ERR: Illegual objective given ({})".format(objective))
        assert (False)

    if do_augment:
        assert(False)
        #aug_images = []
        #for im, mask in zip(images, masks):
        #    im, mask = augment(im, mask, min_size=128, max_angle=20, flip_ratio=0.1, crop_ratio=0.2)
        #    aug_images.append(im)
        #images = np.array(aug_images)

    if reshuffle:
        new_order = np.random.permutation(N)
        images = images[new_order]
        labels = labels[new_order]
        classes = classes[new_order]
        masks  = masks[new_order]
        #print('permutation: {}'.format(new_order[:20]))

    if categorize > 0:
        labels = to_categorical(labels, categorize)

    conf = None
    if rating_confidence is not None:
        conf = np.array([2. + entry[5].shape[0] - np.max(np.max(entry[5], axis=0) - np.min(entry[5], axis=0)) for entry in data]) / 6.
        if reshuffle:
            conf = conf[new_order]

    meta = None
    if return_meta:
        meta = [entry[3] for entry in data]
        if reshuffle:
            meta = reorder(meta, new_order)

    return images, labels, classes, masks, meta, conf


def select_balanced(self, some_set, labels, N, permutation):
    b = some_set[0 == labels][:N]
    m = some_set[1 == labels][:N]
    merged = np.concatenate([b, m], axis=0)
    reshuff = merged[permutation]
    return reshuff


def prepare_data_direct(data, objective='malignancy', rating_scale = 'none', size=None, classes=2, balanced=False, return_meta=False, verbose= 0, reshuffle=True):
    #scale = 'none' if (objective=='malignancy') else "none"
    images, labels, classes, masks, meta, conf = \
        prepare_data(data, objective=objective, categorize=(2 if (objective=='malignancy') else 0), verbose=verbose, reshuffle=reshuffle, return_meta=return_meta, scaling=rating_scale)
    Nb = np.count_nonzero(1 - classes)
    Nm = np.count_nonzero(classes)
    N = np.minimum(Nb, Nm)
    if verbose:
        print("Benign: {}, Malignant: {}".format(Nb, Nm))
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
    return images, labels, classes, masks, meta


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


def select_triplets(elements):
    triplets = [ (t1, t2, t3) for t1, t2, t3 in zip(elements[:-2], elements[1:-1], elements[2:]) ] \
               + [(elements[-2], elements[-1], elements[0]), (elements[-1], elements[0], elements[1])]
    return triplets


def arrange_triplet(elements, labels):
    l2 = lambda a, b: np.sqrt((a - b).dot(a - b))
    trips = [(im[0], im[1], im[2]) if (l2(r[0], r[1]) < l2(r[0], r[2])) else (im[0], im[2], im[1])
             for im, r in zip(elements, labels)]
    return trips


def get_triplet_confidence(labels):
    conf = [(l[0]+l[1]+l[2])/3. for l in labels]
    return conf

def calc_rating_distance_confidence(rating_trips):
    l2 = lambda a, b: np.sqrt((a - b).dot(a - b))
    factor = lambda dp, dn: np.exp(-dp/dn)
    confidence = [factor(l2(r[0], r[1]), l2(r[0], r[2])) for r in rating_trips]
    return confidence

def select_same_pairs(class_A, class_B):
    same = [(a1, a2) for a1, a2 in zip(class_A[:-1], class_A[1:])] + [(class_A[-1], class_A[0])]
    sa_size = len(same)
    same += [(b1, b2) for b1, b2 in zip(class_B[:-1], class_B[1:])] + [(class_B[-1], class_B[0])]
    sb_size = len(same) - sa_size
    assert(len(same) == (len(class_A)+len(class_B)))
    return same, sa_size, sb_size


def prepare_data_siamese(data, objective="malignancy", balanced=False, return_meta=False, verbose= 0):
    if verbose: print('prepare_data_siamese:')
    images, labels, classes, masks, meta, conf = \
        prepare_data(data, categorize=0, objective=objective, return_meta=return_meta, reshuffle=True, verbose=verbose)
    if verbose: print("benign:{}, malignant: {}".format(np.count_nonzero(classes == 0),
                                                        np.count_nonzero(classes == 1)))

    N = images.shape[0]
    benign_filter = np.where(classes == 0)[0]
    malign_filter = np.where(classes == 1)[0]
    M = min(benign_filter.shape[0], malign_filter.shape[0])

    if balanced:
        malign_filter = malign_filter[:M]
        benign_filter = benign_filter[:M]

    #   Handle Patches
    # =========================

    imgs_benign, imgs_malign = images[benign_filter], images[malign_filter]
    different, d_size = select_different_pair(imgs_benign, imgs_malign, n=M)
    same, sb_size, sm_size = select_same_pairs(imgs_benign, imgs_malign)

    image_pairs = same + different
    image_sub1 = np.array([pair[0] for pair in image_pairs])
    image_sub2 = np.array([pair[1] for pair in image_pairs])

    if objective == "malignancy":
        similarity_labels = np.concatenate([np.repeat(0, len(same)),
                                        np.repeat(1, len(different))])
    elif objective == "rating":
        lbls_benign, lbls_malign = labels[benign_filter], labels[malign_filter]
        diff_lbls, d_size = select_different_pair(lbls_benign, lbls_malign, n=M)
        same_lbls, sb_size, sm_size = select_same_pairs(lbls_benign, lbls_malign)

        label_pairs = same_lbls + diff_lbls
        similarity_labels = np.array([np.sqrt((a-b).dot(a-b)) for a, b in label_pairs])
    else:
        print("ERR: {} is not a valid objective".format(objective))
        assert(False)

    #   Handle Masks
    # =========================

    mask_benign, mask_malign = masks[benign_filter], masks[malign_filter]
    different_mask, d = select_different_pair(mask_benign, mask_malign, n=M)
    same_mask, sb, sm = select_same_pairs(mask_benign, mask_malign)
    assert(d == d_size)
    assert ( (sb==sb_size) and (sm==sm_size) )

    mask_pairs = same_mask + different_mask
    mask_sub1 = np.array([pair[0] for pair in mask_pairs])
    mask_sub2 = np.array([pair[1] for pair in mask_pairs])

    #   Handle Meta
    # =========================
    if return_meta:
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
    assert size == image_sub1.shape[0]
    assert size == image_sub2.shape[0]

    # assign confidence classes (weights are resolved online per batch)
    confidence = np.concatenate([  np.repeat('SB', sb_size),
                                   np.repeat('SM', sm_size),
                                   np.repeat('D',  d_size)
                                ])

    if verbose: print("{} pairs of same / {} pairs of different. {} total number of pairs".format(len(same), len(different), size))

    new_order = np.random.permutation(size)

    if return_meta:
        return (    (image_sub1[new_order], image_sub2[new_order]),
                    similarity_labels[new_order],
                    (mask_sub1[new_order], mask_sub2[new_order]),
                    confidence[new_order],
                    (reorder(meta_sub1, new_order), reorder(meta_sub2, new_order))
                )
    else:
        return (    (image_sub1[new_order], image_sub2[new_order]),
                    similarity_labels[new_order],
                    (mask_sub1[new_order], mask_sub2[new_order]),
                    confidence[new_order]
                )


def prepare_data_siamese_simple(data, siamese_rating_factor, objective="malignancy", balanced=False, return_meta=False, verbose=0):
    if verbose:
        print('prepare_data_siamese_simple:')
    images, labels, classes, masks, meta, conf = \
        prepare_data(data, categorize=0, objective=objective, scaling="Scale", return_meta=return_meta, reshuffle=True, verbose=verbose)
    if verbose:
        if return_meta:
            print('Loaded Meta-Data')
        print("benign:{}, malignant: {}".format(np.count_nonzero(classes == 0),
                                                np.count_nonzero(classes == 1)))

    labels *= siamese_rating_factor

    N = images.shape[0]
    #benign_filter = np.where(classes == 0)[0]
    #malign_filter = np.where(classes == 1)[0]
    #M = min(benign_filter.shape[0], malign_filter.shape[0])

    #if balanced:
    #    malign_filter = malign_filter[:M]
    #    benign_filter = benign_filter[:M]

    #   Handle Patches
    # =========================

    #imgs_benign, imgs_malign = images[benign_filter], images[malign_filter]
    #different, d_size = select_different_pair(imgs_benign, imgs_malign, n=M)
    #same, sb_size, sm_size = select_same_pairs(imgs_benign, imgs_malign)

    #image_pairs = same + different
    image_pairs = select_pairs(images)

    image_sub1 = np.array([pair[0] for pair in image_pairs])
    image_sub2 = np.array([pair[1] for pair in image_pairs])

    if objective == "malignancy":
        assert(False)
        #similarity_labels = np.concatenate([np.repeat(0, len(same)),
        #                                np.repeat(1, len(different))])
    elif objective == "rating":
        #lbls_benign, lbls_malign = labels[benign_filter], labels[malign_filter]
        #diff_lbls, d_size = select_different_pair(lbls_benign, lbls_malign, n=M)
        #same_lbls, sb_size, sm_size = select_same_pairs(lbls_benign, lbls_malign)

        #label_pairs = same_lbls + diff_lbls
        label_pairs = select_pairs(labels)
        similarity_labels = np.array([np.sqrt((a-b).dot(a-b)) for a, b in label_pairs])
    else:
        print("ERR: {} is not a valid objective".format(objective))
        assert(False)

    #   Handle Masks
    # =========================

    #mask_benign, mask_malign = masks[benign_filter], masks[malign_filter]
    #different_mask, d = select_different_pair(mask_benign, mask_malign, n=M)
    #same_mask, sb, sm = select_same_pairs(mask_benign, mask_malign)
    #assert(d == d_size)
    #assert ( (sb==sb_size) and (sm==sm_size) )

    #mask_pairs = same_mask + different_mask
    mask_pairs = select_pairs(masks)
    mask_sub1 = np.array([pair[0] for pair in mask_pairs])
    mask_sub2 = np.array([pair[1] for pair in mask_pairs])

    #   Handle Meta
    # =========================
    if return_meta:
        #meta_benign, meta_malign = reorder(meta, benign_filter), reorder(meta, malign_filter)
        #different_meta, d = select_different_pair(meta_benign, meta_malign, n=M)
        #same_meta, sb, sm = select_same_pairs(meta_benign, meta_malign)
        #assert (d == d_size)
        #assert ((sb == sb_size) and (sm == sm_size))

        #meta_pairs = same_meta + different_meta
        meta_pairs = select_pairs(meta)
        meta_sub1 = np.array([pair[0] for pair in meta_pairs])
        meta_sub2 = np.array([pair[1] for pair in meta_pairs])

    #   Final touch
    # =========================

    size = similarity_labels.shape[0]
    assert size == image_sub1.shape[0]
    assert size == image_sub2.shape[0]

    # assign confidence classes (weights are resolved online per batch)
    #confidence = np.concatenate([  np.repeat('SB', sb_size),
    #                               np.repeat('SM', sm_size),
    #                               np.repeat('D',  d_size)
    #                            ])
    confidence = np.repeat('SB', N)

    #if verbose: print("{} pairs of same / {} pairs of different. {} total number of pairs".format(len(same), len(different), size))

    new_order = np.random.permutation(size)

    if return_meta:
        return (    (image_sub1[new_order], image_sub2[new_order]),
                    similarity_labels[new_order],
                    (mask_sub1[new_order], mask_sub2[new_order]),
                    confidence[new_order],
                    (reorder(meta_sub1, new_order), reorder(meta_sub2, new_order))
                )
    else:
        return (    (image_sub1[new_order], image_sub2[new_order]),
                    similarity_labels[new_order],
                    (mask_sub1[new_order], mask_sub2[new_order]),
                    confidence[new_order]
                )


def make_balanced_trip(elements, c1_head, c1_tail, c2_head, c2_tail):
    trips  = [(elements[r], elements[p], elements[n]) for r, p, n in zip(c1_head, c1_tail, c2_head)]
    trips += [(elements[r], elements[p], elements[n]) for r, p, n in zip(c1_tail, c1_head, c2_tail)]
    trips += [(elements[r], elements[p], elements[n]) for r, p, n in zip(c2_head, c2_tail, c1_head)]
    trips += [(elements[r], elements[p], elements[n]) for r, p, n in zip(c2_tail, c2_head, c1_tail)]
    return trips


def prepare_data_triplet(data, objective="malignancy", balanced=False, return_confidence=False, return_meta=False, verbose= 0):
    if verbose:
        print('prepare_data_triplet:')
    images, ratings, classes, masks, meta, conf \
        = prepare_data(data, categorize=0, objective="rating", scaling="Scale", rating_confidence=return_confidence,return_meta=return_meta, reshuffle=True, verbose=verbose)
    if verbose:
        print("benign:{}, malignant: {}".format(np.count_nonzero(classes == 0), np.count_nonzero(classes == 1)))
        if meta is not None: print('Loaded Meta-Data')

    N = images.shape[0]

    if objective=="malignancy":
        benign_filter = np.where(classes == 0)[0]
        malign_filter = np.where(classes == 1)[0]
        M = min(benign_filter.shape[0], malign_filter.shape[0])
        M12 = M // 2
        M   = M12  *2
        malign_filter_a = malign_filter[:M12]
        malign_filter_b = malign_filter[M12:]
        benign_filter_a = benign_filter[:M12]
        benign_filter_b = benign_filter[M12:]

    #   Handle Patches
    # =========================

    if objective=="malignancy":
        image_trips = make_balanced_trip(images, benign_filter_a, benign_filter_b, malign_filter_a, malign_filter_b)
    else:
        image_trips  = select_triplets(images)
        rating_trips = select_triplets(ratings)
        image_trips  = arrange_triplet(image_trips, rating_trips)
    image_sub1 = np.array([pair[0] for pair in image_trips])
    image_sub2 = np.array([pair[1] for pair in image_trips])
    image_sub3 = np.array([pair[2] for pair in image_trips])

    similarity_labels = np.array([0]*N)

    #   Handle Masks
    # =========================

    if objective=="malignancy":
        mask_trips = make_balanced_trip(masks, benign_filter_a, benign_filter_b, malign_filter_a, malign_filter_b)
    else:
        mask_trips = select_triplets(masks)
        mask_trips = arrange_triplet(mask_trips, rating_trips)
    mask_sub1 = np.array([pair[0] for pair in mask_trips])
    mask_sub2 = np.array([pair[1] for pair in mask_trips])
    mask_sub3 = np.array([pair[2] for pair in mask_trips])

    #   Handle Meta
    # =========================
    if return_meta:
        if objective=="malignancy":
            meta_trips = make_balanced_trip(meta, benign_filter_a, benign_filter_b, malign_filter_a, malign_filter_b)
        else:
            meta_trips = select_triplets(meta)
            meta_trips = arrange_triplet(meta_trips, rating_trips)
        meta_sub1 = np.array([pair[0] for pair in meta_trips])
        meta_sub2 = np.array([pair[1] for pair in meta_trips])
        meta_sub3 = np.array([pair[2] for pair in meta_trips])

    #   Final touch
    # =========================

    size = image_sub1.shape[0]
    assert size == mask_sub1.shape[0]

    confidence = np.repeat('SB', N)
    if objective=='rating':
        if return_confidence == "rating":
            conf_trips = select_triplets(conf)
            conf_trips = arrange_triplet(conf_trips, rating_trips)
            confidence = get_triplet_confidence(conf_trips)
            confidence = np.array(confidence)
        elif return_confidence == "rating_distance":
            confidence = calc_rating_distance_confidence(rating_trips)
            confidence = np.array(confidence)

    new_order = np.random.permutation(size)

    if return_meta:
        return (    (image_sub1[new_order], image_sub2[new_order]),
                    similarity_labels[new_order],
                    (mask_sub1[new_order], mask_sub2[new_order]),
                    confidence[new_order],
                    (reorder(meta_sub1, new_order), reorder(meta_sub2, new_order), reorder(meta_sub3, new_order))
                )
    else:
        return (    (image_sub1[new_order], image_sub2[new_order], image_sub3[new_order]),
                    similarity_labels[new_order],
                    (mask_sub1[new_order], mask_sub2[new_order], mask_sub3[new_order]),
                    confidence[new_order]
                )
