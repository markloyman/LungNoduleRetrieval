import numpy as np
try:
    from Network.Common import prepare_data
except:
    from Common import prepare_data


def select_triplets(elements):
    triplets = [ (t1, t2, t3) for t1, t2, t3 in zip(elements[:-2], elements[1:-1], elements[2:]) ] \
               + [(elements[-2], elements[-1], elements[0]), (elements[-1], elements[0], elements[1])]
    return triplets


def check_triplet_order(ratings_triplet, rating_distance):
    rank_status = [rating_distance(r0, r_pos) < rating_distance(r0, r_neg) for r0, r_pos, r_neg in ratings_triplet]
    rank_status = np.array(rank_status)
    return rank_status  # true if no need to flip


def arrange_triplet(elements, labels):
    trips = [(im[0], im[1], im[2]) if lbl else (im[0], im[2], im[1])
             for im, lbl in zip(elements, labels)]
    return trips


def get_triplet_confidence(labels):
    conf = [(l[0]+l[1]+l[2])/3. for l in labels]
    return conf


def calc_rating_distance_confidence(rating_trips):
    l2 = lambda a, b: np.sqrt((a - b).dot(a - b))
    factor = lambda dp, dn: np.exp(-dp/dn)
    confidence = [factor(l2(r[0], r[1]), l2(r[0], r[2])) for r in rating_trips]
    return confidence


def make_balanced_trip(elements, c1_head, c1_tail, c2_head, c2_tail):
    trips  = [(elements[r], elements[p], elements[n]) for r, p, n in zip(c1_head, c1_tail, c2_head)]
    trips += [(elements[r], elements[p], elements[n]) for r, p, n in zip(c1_tail, c1_head, c2_tail)]
    trips += [(elements[r], elements[p], elements[n]) for r, p, n in zip(c2_head, c2_tail, c1_head)]
    trips += [(elements[r], elements[p], elements[n]) for r, p, n in zip(c2_tail, c2_head, c1_tail)]
    return trips


def prepare_data_triplet(data, objective="malignancy", rating_distance="mean", balanced=False, return_confidence=False, return_meta=False, verbose= 0):
    if verbose:
        print('prepare_data_triplet:')
    images, ratings, classes, masks, meta, conf, nod_size, _ \
        = prepare_data(data, rating_format="raw", scaling="none", reshuffle=True, verbose=verbose)

    N = images.shape[0]

    if balanced:
        print('Create a balanced split')
        benign_filter = np.where(classes == 0)[0]
        malign_filter = np.where(classes == 1)[0]
        M = min(benign_filter.shape[0], malign_filter.shape[0])
        M12 = M // 2
        M   = M12 *2
        malign_filter_a = malign_filter[:M12]
        malign_filter_b = malign_filter[M12:]
        benign_filter_a = benign_filter[:M12]
        benign_filter_b = benign_filter[M12:]
    else:
        rating_trips = select_triplets(ratings)
        distance = l2_distance if rating_distance == 'mean' else cluster_distance
        trip_rank_status = check_triplet_order(rating_trips, rating_distance=distance)

    #   Handle Patches
    # =========================

    if balanced:
        image_trips = make_balanced_trip(images, benign_filter_a, benign_filter_b, malign_filter_a, malign_filter_b)
    else:
        image_trips  = select_triplets(images)
        image_trips  = arrange_triplet(image_trips, trip_rank_status)
    image_sub1 = np.array([pair[0] for pair in image_trips])
    image_sub2 = np.array([pair[1] for pair in image_trips])
    image_sub3 = np.array([pair[2] for pair in image_trips])

    similarity_labels = np.array([0]*N)

    #   Handle Masks
    # =========================

    if balanced:
        mask_trips = make_balanced_trip(masks, benign_filter_a, benign_filter_b, malign_filter_a, malign_filter_b)
    else:
        mask_trips = select_triplets(masks)
        mask_trips = arrange_triplet(mask_trips, trip_rank_status)
    mask_sub1 = np.array([pair[0] for pair in mask_trips])
    mask_sub2 = np.array([pair[1] for pair in mask_trips])
    mask_sub3 = np.array([pair[2] for pair in mask_trips])

    #   Handle Meta
    # =========================
    if return_meta:
        if balanced:
            meta_trips = make_balanced_trip(meta, benign_filter_a, benign_filter_b, malign_filter_a, malign_filter_b)
        else:
            meta_trips = select_triplets(meta)
            meta_trips = arrange_triplet(meta_trips, trip_rank_status)
        meta_sub1 = np.array([pair[0] for pair in meta_trips])
        meta_sub2 = np.array([pair[1] for pair in meta_trips])
        meta_sub3 = np.array([pair[2] for pair in meta_trips])

    #   Final touch
    # =========================

    size = image_sub1.shape[0]
    assert M*2 == size

    confidence = np.repeat('SB', size)
    if objective=='rating':
        if return_confidence == "rating":
            conf_trips = select_triplets(conf)
            conf_trips = arrange_triplet(conf_trips, trip_rank_status)
            confidence = get_triplet_confidence(conf_trips)
            confidence = np.array(confidence)
        elif return_confidence == "rating_distance":
            confidence = calc_rating_distance_confidence(trip_rank_status)
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
