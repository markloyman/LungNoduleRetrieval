import pickle
import numpy as np
import random
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


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
        plt.hist(images, bins=50)

    if window is not None:
        images = np.clip(images, window[0], window[1])

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


def generate_nodule_dataset(filename, test_ratio, validation_ratio, window=(-1000, 350)):
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

    trainData  = normalize_all(trainData, mean, std, window=window)
    testData   = normalize_all(testData,  mean, std, window=window)
    validData  = normalize_all(validData, mean, std, window=window)

    getImageStatistics(trainData,   verbose=True)
    getImageStatistics(validData,   verbose=True)
    getImageStatistics(testData,    verbose=True)

    pickle.dump((testData, validData, trainData), open('Dataset.p','bw'))

    #testData  = [ (entry['patch'], entry['mask'], entry['label'], entry['info']) for entry in testData ]
    #validData = [ (entry['patch'], entry['mask'], entry['label'], entry['info']) for entry in validData]
    #trainData = [ (entry['patch'], entry['mask'], entry['label'], entry['info']) for entry in trainData]

    #pickle.dump((testData, validData, trainData), open('Dataset.p', 'bw'))

    plt.show()

    return testData, validData, trainData

# =========================
#   Load
# =========================

def load_nodule_dataset():
    testData, validData, trainData = pickle.load(open('Dataset.p', 'br'))

    testData  = [ (entry['patch'], entry['mask'], entry['label'], entry['info']) for entry in testData ]
    validData = [ (entry['patch'], entry['mask'], entry['label'], entry['info']) for entry in validData]
    trainData = [ (entry['patch'], entry['mask'], entry['label'], entry['info']) for entry in trainData]

    return testData, validData, trainData


def load_nodule_raw_dataset():
    testData, validData, trainData = pickle.load(open('RawDataset.p', 'br'))
    return testData, validData, trainData


# =========================
#   Prepare Data
# =========================

def prepare_data(data, classes=0, size=0, categorize=True, return_meta=False, reshuffle=False, verbose = 0):
    # Entry:
    # 0 'patch'
    # 1 'mask'
    # 2 'label'
    # 3 'info'

    N = len(data)
    old_size =  data[0][0].shape
    images = np.array([resize(entry[0], (size, size), preserve_range=True) for entry in data]).reshape(N,size,size,1)
    labels = np.array([entry[2] for entry in data]).reshape(N,1)

    if verbose: print("Image size changed from {} to {}".format(old_size, images[0].shape))

    if reshuffle:
        new_order = np.random.permutation(N)
        images = images[new_order]
        labels = labels[new_order]

    if categorize:
        labels = to_categorical(labels, classes)

    if return_meta:
        meta = [entry[3] for entry in data]
        if reshuffle:
            meta = meta[new_order]
        return images, labels, meta
    else:
        return images, labels


def prepare_data_siamese(data, size, return_meta=False, verbose = 0):
    if return_meta:
        images, labels, meta = prepare_data(data, size=size, categorize=False, return_meta=True, reshuffle=True, verbose=verbose)
        if verbose: print('Loaded Meta-Data')
    else:
        images, labels = prepare_data(data, size=size, categorize=False, return_meta=False, reshuffle=True, verbose=verbose)
    if verbose: print("benign:{}, malignant: {}".format(    np.count_nonzero(labels==0),
                                                np.count_nonzero(labels==1)))

    N = images.shape[0]

    imgs_benign = np.split(images[np.where(labels == 0)[0]], [np.floor(N / 4).astype('uint')])
    imgs_malign = np.split(images[np.where(labels == 1)[0]], [np.floor(N / 4).astype('uint')])

    #lbls_benign = np.split(labels[np.where(labels == 0)[0]], [np.floor(N / 4).astype('uint')])
    #lbls_malign = np.split(labels[np.where(labels == 1)[0]], [np.floor(N / 4).astype('uint')])

    if return_meta:
        meta_benign = np.split(meta[np.where(labels == 0)[0]], [np.floor(N / 4).astype('uint')])
        meta_malign = np.split(meta[np.where(labels == 1)[0]], [np.floor(N / 4).astype('uint')])

    different = [(b,m) for b,m in zip(imgs_benign[0], imgs_malign[0])]
    #different_lbls = [(0, 1)] * len(different)


    bN = np.floor(imgs_benign[1].shape[0] / 2).astype('uint')
    mN = np.floor(imgs_malign[1].shape[0] / 2).astype('uint')
    same = [(first, last) for first, last in zip(imgs_benign[1][:bN], imgs_benign[1][bN:bN+bN])]
    #same_lbls = [(0, 0)] * len(same)
    same = same + [(first, last) for first, last in zip(imgs_malign[1][:mN], imgs_malign[1][mN:mN+mN])]
    #same_lbls = same_lbls + [(1, 1)] * (len(same)-len(same_lbls))

    image_pairs       = same + different
    #label_pairs       = same_lbls + different_lbls

    similarity_labels = np.concatenate([np.repeat(0, len(same)),
                                        np.repeat(1, len(different))])


    image_sub1 = np.array([pair[0] for pair in image_pairs])
    image_sub2 = np.array([pair[1] for pair in image_pairs])

    if return_meta:
        different_meta = [(b, m) for b, m in zip(meta_benign[0], meta_malign[0])]
        same_meta = [(first, last) for first, last in zip(meta_benign[1][:bN], meta_benign[1][bN:bN + bN])]
        same_meta = same_meta + [(first, last) for first, last in zip(meta_malign[1][:mN], meta_malign[1][mN:mN + mN])]
        meta_pairs = different_meta + same_meta

        meta_sub1 = np.array([pair[0] for pair in meta_pairs])
        meta_sub2 = np.array([pair[1] for pair in meta_pairs])

    size = similarity_labels.shape[0]
    assert size == image_sub1.shape[0]
    assert size == image_sub2.shape[0]

    if verbose: print("{} pairs of same / {} pairs of different. {} total number of pairs".format(len(same), len(different),size))

    new_order = np.random.permutation(size)

    if return_meta:
        return ((image_sub1[new_order], image_sub2[new_order]), similarity_labels[new_order], (meta_sub1[new_order], meta_sub2[new_order]))
    else:
        return ((image_sub1[new_order], image_sub2[new_order]), similarity_labels[new_order])


def prepare_data_siamese_overlap(data, size, return_meta=False, verbose = 0):
    if return_meta:
        images, labels, meta = prepare_data(data, size=size, categorize=False, return_meta=True, reshuffle=True, verbose=verbose)
        if verbose: print('Loaded Meta-Data')
    else:
        images, labels = prepare_data(data, size=size, categorize=False, return_meta=False, reshuffle=True, verbose=verbose)
    if verbose: print("benign:{}, malignant: {}".format(    np.count_nonzero(labels==0),
                                                np.count_nonzero(labels==1)))

    N = images.shape[0]
    #imgs_benign = np.split(images[np.where(labels == 0)[0]], [np.floor(N / 4).astype('uint')])
    imgs_benign = images[np.where(labels == 0)[0]]
    #imgs_malign = np.split(images[np.where(labels == 1)[0]], [np.floor(N / 4).astype('uint')])
    imgs_malign = images[np.where(labels == 1)[0]]
    M = min(imgs_benign.shape[0], imgs_malign.shape[0])

    #lbls_benign = np.split(labels[np.where(labels == 0)[0]], [np.floor(N / 4).astype('uint')])
    #lbls_malign = np.split(labels[np.where(labels == 1)[0]], [np.floor(N / 4).astype('uint')])

    if return_meta:
        meta_benign = meta[np.where(labels == 0)[0]]
        meta_malign = meta[np.where(labels == 1)[0]]

    different = [(b,m) for b,m in zip(imgs_benign[:M], imgs_malign[:M])]
    #different_lbls = [(0, 1)] * len(different)
    if verbose: print("{} pairs of different".format(len(different)))

    bN = np.floor(imgs_benign.shape[0] / 2).astype('uint')
    mN = np.floor(imgs_malign.shape[0] / 2).astype('uint')
    same = [(first, last) for first, last in zip(imgs_benign[:bN], imgs_benign[bN:bN+bN])]
    #same_lbls = [(0, 0)] * len(same)
    same = same + [(first, last) for first, last in zip(imgs_malign[:mN], imgs_malign[mN:mN+mN])]
    #same_lbls = same_lbls + [(1, 1)] * (len(same)-len(same_lbls))
    if verbose: print("{} pairs of same".format(len(same)))

    image_pairs       = same + different
    #label_pairs       = same_lbls + different_lbls

    similarity_labels = np.concatenate([np.repeat(0, len(same)),
                                        np.repeat(1, len(different))])


    image_sub1 = np.array([pair[0] for pair in image_pairs])
    image_sub2 = np.array([pair[1] for pair in image_pairs])

    if return_meta:
        different_meta = [(b, m) for b, m in zip(meta_benign[:M], meta_malign[:M])]
        same_meta = [(first, last) for first, last in zip(meta_benign[:bN], meta_benign[bN:bN + bN])]
        same_meta = same_meta + [(first, last) for first, last in zip(meta_malign[:mN], meta_malign[mN:mN + mN])]
        meta_pairs = different_meta + same_meta

        meta_sub1 = np.array([pair[0] for pair in meta_pairs])
        meta_sub2 = np.array([pair[1] for pair in meta_pairs])

    size = similarity_labels.shape[0]
    assert size == image_sub1.shape[0]
    assert size == image_sub2.shape[0]

    if verbose: print("{} number of pairs".format(size))
    new_order = np.random.permutation(size)

    if return_meta:
        return ((image_sub1[new_order], image_sub2[new_order]), similarity_labels[new_order], (meta_sub1[new_order], meta_sub2[new_order]))
    else:
        return ((image_sub1[new_order], image_sub2[new_order]), similarity_labels[new_order])


def prepare_data_siamese_chained(data, size, return_meta=False, verbose = 0, weighted_samples = False):
    if return_meta:
        images, labels, meta = prepare_data(data, size=size, categorize=False, return_meta=True, reshuffle=True, verbose=verbose)
        if verbose: print('Loaded Meta-Data')
    else:
        images, labels = prepare_data(data, size=size, categorize=False, return_meta=False, reshuffle=True, verbose=verbose)
    if verbose: print("benign:{}, malignant: {}".format(    np.count_nonzero(labels==0),
                                                np.count_nonzero(labels==1)))

    N = images.shape[0]
    imgs_benign = images[np.where(labels == 0)[0]]
    imgs_malign = images[np.where(labels == 1)[0]]
    M = min(imgs_benign.shape[0], imgs_malign.shape[0])

    if return_meta:
        meta_benign = meta[np.where(labels == 0)[0]]
        meta_malign = meta[np.where(labels == 1)[0]]

    different = [(b,m) for b,m in zip(
                        np.concatenate([[b, m] for b, m in zip(imgs_benign[:M], imgs_benign[:M])])[:-1],
                        np.concatenate([[b, m] for b, m in zip(imgs_malign[:M], imgs_malign[:M])])[1:])
                 ] + [(imgs_benign[:M][-1], imgs_malign[0])]
    d_size = len(different)

    same =        [(first, last) for first, last in
                        zip(imgs_benign[:-1], imgs_benign[1:])] + [(imgs_benign[0], imgs_benign[-1])]
    sb_size = len(same)

    same = same + [(first, last) for first, last in
                        zip(imgs_malign[:-1], imgs_malign[1:])] + [(imgs_malign[0], imgs_malign[-1])]
    sm_size = len(same) - sb_size

    image_pairs       = same + different
    similarity_labels = np.concatenate([np.repeat(0, len(same)),
                                        np.repeat(1, len(different))])

    image_sub1 = np.array([pair[0] for pair in image_pairs])
    image_sub2 = np.array([pair[1] for pair in image_pairs])

    if return_meta:
        different_meta = [(b,m) for b,m in zip(
                        np.concatenate([[b, m] for b, m in zip(meta_benign[:M], meta_benign[:M])])[:-1],
                        np.concatenate([[b, m] for b, m in zip(meta_malign[:M], meta_malign[:M])])[1:])
                 ] + [(meta_benign[:M][-1], meta_malign[0])]

        same_meta = [(first, last) for first, last in zip(meta_benign[:-1], meta_benign[1:])] + [(meta_benign[0], meta_benign[-1])]
        same_meta = same_meta + [(first, last) for first, last in zip(meta_malign[:-1], meta_malign[1:])] + [(meta_malign[0], meta_malign[-1])]
        meta_pairs = different_meta + same_meta

        meta_sub1 = np.array([pair[0] for pair in meta_pairs])
        meta_sub2 = np.array([pair[1] for pair in meta_pairs])

    size = similarity_labels.shape[0]
    assert size == image_sub1.shape[0]
    assert size == image_sub2.shape[0]

    if weighted_samples:
        sb_w = np.round(size / (4 * sb_size), 1)
        sm_w = np.round(size / (4 * sm_size), 1)
        d_w  = np.round(size / (2 * d_size),  1)
        confidence = np.concatenate([  np.repeat(sb_w, sb_size),
                                       np.repeat(sm_w, sm_size),
                                       np.repeat(d_w,  d_size)
                                    ])
        if verbose: print('Class Weights: Benign {:.2f}, Malig {:.2f}, Different {:.2f}'.format(sb_w, sm_w, d_w))

    if verbose: print("{} pairs of same / {} pairs of different. {} total number of pairs".format(len(same), len(different), size))

    new_order = np.random.permutation(size)

    if weighted_samples:
        if return_meta:
            return ((image_sub1[new_order], image_sub2[new_order]), similarity_labels[new_order], confidence[new_order], (meta_sub1[new_order], meta_sub2[new_order]))
        else:
            return ((image_sub1[new_order], image_sub2[new_order]), similarity_labels[new_order], confidence[new_order])
    else:
        if return_meta:
            return ((image_sub1[new_order], image_sub2[new_order]), similarity_labels[new_order], None, (meta_sub1[new_order], meta_sub2[new_order]))
        else:
            return ((image_sub1[new_order], image_sub2[new_order]), similarity_labels[new_order], None)


if __name__ == "__main__":

    generate_nodule_dataset(    filename='LIDC/NodulePatchesCliqueByMalignancy.p',
                                test_ratio=0.2,
                                validation_ratio=0.25)

    plt.show()