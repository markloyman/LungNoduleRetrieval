import pickle
import numpy as np
import random
from skimage.transform import resize
from scipy.misc import imresize
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from Network.dataUtils import augment

# =========================
#   Generate Nodule Dataset
# =========================


def calc_split_size(N, test_ratio, validation_ratio):
    testN = np.floor(N * test_ratio).astype('uint')
    validN = np.floor((N - testN) * validation_ratio).astype('uint')
    trainN = N - testN - validN

    assert trainN > 0
    print("Size- test:{}-{:.2f}, valid:{}-{:.2f}, train:{}-{:.2f}".format(testN, testN/N, validN, validN/N, trainN, trainN/N))

    return (testN, validN, trainN)


def split_dataset(data, testN, validN, trainN):
    random.shuffle(data)
    shift = [0, testN, testN + validN]

    testData  = data[shift[0]:shift[0] + testN]
    validData = data[shift[1]:shift[1] + validN]
    trainData = data[shift[2]:shift[2] + trainN]

    assert testN  == len(testData)
    assert validN == len(validData)
    assert trainN == len(trainData)

    return (testData, validData, trainData)


def getImageStatistics(data, window=None, verbose=False):
    images = np.array([entry['patch'] for entry in data]).flatten()

    if verbose:
        plt.figure()
        plt.hist(images, bins=500)

    if window is not None:
        images = np.clip(images, window[0], window[1])
        #images = images[(images>=window[0])*(images<=window[1])]

    mean   = np.mean(images)
    std    = np.std(images)

    return mean, std


def normalize(image, mean, std, window=None):
    assert std > 0
    if window is not None:
        image = np.clip(image, window[0], window[1])
    image = image - mean
    image = image.astype('float') / std

    return image


def normalize_all(dataset, mean=0, std=1, window=None):
    new_dataset = []
    for entry in dataset:
        entry['patch'] = normalize(entry['patch'], mean, std, window)
        new_dataset.append(entry)
    return new_dataset


def uniform(image, mean=0, window=None, centered=True):
    MIN_BOUND, MAX_BOUND = window
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image = np.clip(image, 0.0, 1.0)
    if centered:
        mean = (mean - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image -= mean
    return image


def uniform_all(dataset, mean=0, window=None, centered=True):
    new_dataset = []
    for entry in dataset:
        entry['patch'] = uniform(entry['patch'], mean, window, centered=centered)
        new_dataset.append(entry)
    return new_dataset


def generate_nodule_dataset(filename, test_ratio, validation_ratio, window=(-1000, 400), normalize='Normal', dump=True, output_filename='Dataset.p'):
    #   size of test set        N*test_ratio
    #   size of validation set  N*(1-test_ratio)*validation_ratio
    #   size of training set    N*test_ratio

    M, B, U = pickle.load(open(filename, 'br'))

    print("Split Malignancy Set. Loaded total of {} images".format(len(M)))
    testM, validM, trainM = calc_split_size(len(M), test_ratio, validation_ratio)
    print("Split Benign Set. Loaded total of {} images".format(len(B)))
    testB, validB, trainB = calc_split_size(len(B), test_ratio, validation_ratio)
    print("Ignoring {} unknown images".format(len(U)))

    testDataM, validDataM, trainDataM = split_dataset(M, testM, validM, trainM)
    testDataB, validDataB, trainDataB = split_dataset(B, testB, validB, trainB)

    # group by designation (test,validation,training]
    testData  = np.hstack([testDataM,  testDataB])
    validData = np.hstack([validDataM, validDataB])
    trainData = np.hstack([trainDataM, trainDataB])

    random.shuffle(testData)
    random.shuffle(validData)
    random.shuffle(trainData)

    mean, std = getImageStatistics(trainData, window=window, verbose=True)
    print('Training Statistics: Mean {:.2f} and STD {:.2f}'.format(mean, std))

    if normalize is 'Uniform':
        trainData = uniform_all(trainData, mean,  window=window, centered=True)
        testData  = uniform_all(testData,  mean,  window=window, centered=True)
        validData = uniform_all(validData, mean,  window=window, centered=True)
    elif normalize is 'UniformNC':
        trainData = uniform_all(trainData, mean, window=window, centered=False)
        testData  = uniform_all(testData, mean, window=window, centered=False)
        validData = uniform_all(validData, mean, window=window, centered=False)
    elif normalize is 'Normal':
        trainData  = normalize_all(trainData, mean, std, window=window)
        testData   = normalize_all(testData,  mean, std, window=window)
        validData  = normalize_all(validData, mean, std, window=window)

    getImageStatistics(trainData,   verbose=True)
    getImageStatistics(validData,   verbose=True)
    getImageStatistics(testData,    verbose=True)

    if dump:
        pickle.dump((testData, validData, trainData), open(output_filename, 'bw'))
        print('Dumped')
    else:
        print('No Dump')

    plt.show()

    return testData, validData, trainData

# =========================
#   Load
# =========================


def load_nodule_dataset(size=128, res=1.0, apply_mask_to_patch=False, sample='Normal'):

    if res is 'Legacy':
        filename = 'Dataset{:.0f}-{}-{}.p'.format(size, res, sample)
    else:
        filename = 'Dataset{:.0f}-{:.1f}-{}.p'.format(size, res, sample)

    testData, validData, trainData = pickle.load(open(filename, 'br'))
    print('Loaded: {}'.format(filename))
    print("\tImage Range = [{:.1f}, {:.1f}]".format(np.max(trainData[0]['patch']), np.min(trainData[0]['patch'])))
    print("\tMasks Range = [{}, {}]".format(np.max(trainData[0]['mask']), np.min(trainData[0]['mask'])))
    #print("\tLabels Range = [{}, {}]".format(np.max(trainData[0]['label']), np.min(trainData[0]['label'])))

    if apply_mask_to_patch:
        print('WRN: apply_mask_to_patch is for debug only')
        testData  = [(entry['patch']*(0.3+0.7*entry['mask']), entry['mask'], entry['label'], entry['info']) for entry in testData]
        validData = [(entry['patch']*(0.3+0.7*entry['mask']), entry['mask'], entry['label'], entry['info']) for entry in validData]
        trainData = [(entry['patch']*(0.3+0.7*entry['mask']), entry['mask'], entry['label'], entry['info']) for entry in trainData]
    else:
        testData  = [ (entry['patch'], entry['mask'], entry['label'], entry['info']) for entry in testData ]
        validData = [ (entry['patch'], entry['mask'], entry['label'], entry['info']) for entry in validData]
        trainData = [ (entry['patch'], entry['mask'], entry['label'], entry['info']) for entry in trainData]

    return testData, validData, trainData


def load_nodule_raw_dataset(size=128, res=1.0, sample='Normal'):
    if res is 'Legacy':
        filename = 'Dataset{:.0f}-{}-{}.p'.format(size, res, sample)
    else:
        filename = 'Dataset{:.0f}-{:.1f}-{}.p'.format(size, res, sample)
    testData, validData, trainData = pickle.load(open(filename, 'br'))
    return testData, validData, trainData


# =========================
#   Prepare Data
# =========================


def reorder(a_list, order):
    return [a_list[order[i]] for i in range(len(order))]


def prepare_data(data, classes=0, new_size=None, do_augment=False, categorize=True, return_meta=False, reshuffle=False, verbose = 0):
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
    labels = np.array([entry[2] for entry in data]).reshape(N, 1)

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
        masks  = masks[new_order]
        #print('permutation: {}'.format(new_order[:20]))

    if categorize:
        labels = to_categorical(labels, classes)

    if return_meta:
        meta = [entry[3] for entry in data]
        if reshuffle:
            meta = reorder(meta, new_order)
        return images, labels, masks, meta
    else:
        return images, labels, masks


def select_different_pair(class_A, class_B, n):
    different = [(a, b) for a, b in zip(
        np.concatenate( [[a, a] for a in class_A[:n]] )[:-1],
        np.concatenate( [[b, b] for b in class_B[:n]] )[1:])
                 ] + [(class_A[:n][-1], class_B[0])]
    assert(2*n == len(different))
    return different, len(different)


def select_same_pairs(class_A, class_B):
    same = [(a1, a2) for a1, a2 in zip(class_A[:-1], class_A[1:])] + [(class_A[-1], class_A[0])]
    sa_size = len(same)
    same += [(b1, b2) for b1, b2 in zip(class_B[:-1], class_B[1:])] + [(class_B[-1], class_B[0])]
    sb_size = len(same) - sa_size
    assert(len(same) == (len(class_A)+len(class_B)))
    return same, sa_size, sb_size


def prepare_data_siamese(data, size, return_meta=False, verbose= 0):
    if verbose: print('prepare_data_siamese:')
    if return_meta:
        images, labels, masks, meta = \
            prepare_data(data, categorize=False, return_meta=True, reshuffle=True, verbose=verbose)
        if verbose: print('Loaded Meta-Data')
    else:
        images, labels, masks = \
            prepare_data(data, categorize=False, return_meta=False, reshuffle=True, verbose=verbose)
    if verbose: print("benign:{}, malignant: {}".format(np.count_nonzero(labels == 0),
                                                        np.count_nonzero(labels == 1)))

    N = images.shape[0]
    benign_filter = np.where(labels == 0)[0]
    malign_filter = np.where(labels == 1)[0]
    M = min(benign_filter.shape[0], malign_filter.shape[0])

    #   Handle Patches
    # =========================

    imgs_benign, imgs_malign = images[benign_filter], images[malign_filter]
    different, d_size = select_different_pair(imgs_benign, imgs_malign, n=M)
    same, sb_size, sm_size = select_same_pairs(imgs_benign, imgs_malign)

    image_pairs = same + different
    image_sub1 = np.array([pair[0] for pair in image_pairs])
    image_sub2 = np.array([pair[1] for pair in image_pairs])

    similarity_labels = np.concatenate([np.repeat(0, len(same)),
                                        np.repeat(1, len(different))])

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
        meta_sub1 = np.array([pair[0] for pair in meta_pairs])
        meta_sub2 = np.array([pair[1] for pair in meta_pairs])

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

if __name__ == "__main__":
    random.seed(1337)
    np.random.seed(1337)  # for reproducibility

    generate_nodule_dataset(filename='LIDC/NodulePatches144-0.7ByMalignancy.p',
                            output_filename='Dataset144-0.7-Normal.p',
                            test_ratio=0.2,
                            validation_ratio=0.25,
                            window=(-1000, 400),
                            normalize='Normal',
                            dump=True)

    '''
    generate_nodule_dataset(filename='LIDC/NodulePatches144-0.7ByMalignancy.p',
                            output_filename='Dataset144-0.7-Normal.p',
                            test_ratio=0.2,
                            validation_ratio=0.25,
                            window=(-1000, 400),
                            normalize='Normal',
                            dump=True)
    '''
    plt.show()

'''

    generate_nodule_dataset(filename='LIDC/NodulePatches144-LegacyByMalignancy.p',
                            output_filename='Dataset144-Legacy-Normal.p',
                            test_ratio=0.2,
                            validation_ratio=0.25,
                            window=(-1000, 400),
                            uniform_normalize=False,
                            dump=True)

    generate_nodule_dataset(filename='LIDC/NodulePatches144-LegacyByMalignancy.p',
                            output_filename='Dataset144-Legacy-Uniform.p',
                            test_ratio=0.2,
                            validation_ratio=0.25,
                            window=(-1000, 400),
                            uniform_normalize=True,
                            dump=True)
'''
