import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from functools import reduce

try:
    from Network.dataUtils import rating_normalize, crop_center
except:
    from dataUtils import rating_normalize, crop_center


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

    if window is not None:
        #images_ = np.clip(images, window[0], window[1])
        images_ = images[(images > window[0]) & (images < window[1])]
    else:
        images_ = images
        #images = images[(images>=window[0])*(images<=window[1])]

    if verbose:
        plt.figure()
        plt.subplot(211)
        plt.hist(images, bins=500)
        plt.subplot(212)
        plt.hist(images_, bins=500)

    mean   = np.mean(images_)
    std    = np.std(images_)

    return mean, std


def normalize(image, mean, std, window=None):
    image_n = image
    if window is not None:
        image_n = np.clip(image_n, window[0], window[1])
    image_n = image_n - mean
    image_n = image_n.astype('float') / std

    return image_n


def normalize_all(dataset, mean=0, std=1, window=None):
    #new_dataset = []
    for entry in dataset:
        entry['patch'] = normalize(entry['patch'], mean, std, window)
        #new_dataset.append(entry)
    return dataset


def uniform(image, mean=0, window=None, centered=True):
    MIN_BOUND, MAX_BOUND = window
    image_u = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image_u = np.clip(image_u, 0.0, 1.0)
    if centered:
        mean = (mean - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image_u -= mean
    return image_u


def uniform_all(dataset, mean=0, window=None, centered=True):
    new_dataset = []
    for entry in dataset:
        entry['patch'] = uniform(entry['patch'], mean, window, centered=centered)
        new_dataset.append(entry)
    return new_dataset


def split_to_crossvalidation_groups(dataset, n_groups):
    subsets = [[] for i in range(n_groups)]
    benign_count = [0] * n_groups
    malig_count   = [0]*n_groups
    unknown_count = [0]*n_groups

    data_by_patients = {}
    for entry in dataset:
        patient_id = entry['info'][0]
        if patient_id in data_by_patients.keys():
            data_by_patients[patient_id] += [entry]
        else:
            data_by_patients[patient_id] = [entry]

    for patient_id, entry_list in data_by_patients.items():
        label_count = np.bincount(np.array([entry['label'] for entry in entry_list]), minlength=3)
        max_size_per_group = [np.max([b + label_count[0], m + label_count[1], u + label_count[2]]) for b, m, u in
                                zip(benign_count, malig_count, unknown_count)]
        group_id = np.argmin(max_size_per_group)
        '''
        majority_class = np.argmax(label_count)
        if majority_class == 0: # benign
            group_id = np.argmin(benign_count)
        elif majority_class == 1: # malignancy
            group_id = np.argmin(malig_count)
        elif majority_class == 2: # unknown
            group_id = np.argmin(unknown_count)
        else:
            assert(False)
        '''
        subsets[group_id] += entry_list
        benign_count[group_id] += label_count[0]
        malig_count[group_id] += label_count[1]
        unknown_count[group_id] += label_count[2]

    for i, group in enumerate(subsets):
        count = len(group)
        print("Group #{}: {} entries (b:{}, m:{}, u:{})".format(i, count, benign_count[i], malig_count[i], unknown_count[i]))

    return subsets


def scale_image_values(dataset, window=(-1000, 400), statistics=None, normalize='Normal'):
    if statistics is None:
        mean, std = getImageStatistics(dataset, window=window, verbose=True)
    else:
        getImageStatistics(dataset, verbose=True)
        mean, std = statistics
    print('Training Statistics: Mean {:.2f} and STD {:.2f}'.format(mean, std))

    if normalize is 'Uniform':
        dataset = uniform_all(dataset, mean, window=window, centered=True)
    elif normalize is 'UniformNC':
        dataset = uniform_all(dataset, mean, window=window, centered=False)
    elif normalize is 'Normal':
        dataset = normalize_all(dataset, mean, std, window=window)

    getImageStatistics(dataset, verbose=True)

    return dataset


def crop_dataset(dataset, size):
    for entry in dataset:
        patch, mask = crop_center(entry['patch'], entry['mask'], size=size)
        entry['patch'] = patch
        entry['mask']  = mask
    return dataset


def filter_to_primary(dataset):
    def get_filtered_iterator():
        return filter(lambda entry: np.max(entry['weights']) == 1, dataset)
    cluster_ids = np.array([entry['info'][0][-4:] + reduce(lambda x, y: x + y, entry['info'][-1]) for entry in get_filtered_iterator()])
    cluster_ids_unique_map = np.unique(cluster_ids, return_inverse=True)[1]
    weights = np.array([np.sum(entry['weights']) for entry in get_filtered_iterator()])

    selection = []
    for c_id in np.unique(cluster_ids_unique_map):
        cluster_weights = weights[c_id == cluster_ids_unique_map]
        cluster_indices = np.argwhere(c_id == cluster_ids_unique_map)
        selection += [cluster_indices[np.argmax(cluster_weights)]]
    selection = np.concatenate(selection)

    subset = list(get_filtered_iterator())
    dataset = [subset[i] for i in selection]

    return dataset


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