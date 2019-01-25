import pickle
import numpy as np
from functools import partial, reduce
from skimage import transform
from config import dataset_dir as input_dir, output_dir
from Network.FileManager import Dataset, Dataset3d, DatasetFromPredication
from experiments import CrossValidationManager
from Network.dataUtils import rating_normalize, crop_center, l2_distance
from Network.dataUtils import rating_clusters_distance, rating_clusters_distance_matrix, reorder


# =========================
#   Load
# =========================

def build_loader(size=128, res=1.0, apply_mask_to_patch=False, sample='Normal', dataset_type='Clean',
                 configuration=None, n_groups=5, config_name='LEGACY', run=None, epoch=None):

    if config_name == 'LEGACY':
        loader = partial(load_nodule_dataset, size, res, apply_mask_to_patch, sample, configuration, n_groups, dataset_type)
    elif config_name == 'PRED':
        loader = partial(load_nodule_dataset_pred, size, res, apply_mask_to_patch, sample, configuration)
    elif config_name == 'RET':
        loader = partial(load_nodule_dataset_ret, run, epoch, apply_mask_to_patch, configuration)
    else:
        print('{} - illegal config_name'.format(config_name))
        assert False

    return loader


def build_loader_3d(configuration, net_type, run, epoch, n_groups=5):
    loader = partial(load_nodule_dataset_3d, configuration, net_type, run, epoch, n_groups)
    return loader


def load_nodule_dataset_3d(configuration, net_type, run, epoch, n_groups=5):

    DataFile = Dataset3d(configuration, dir=input_dir)
    trainData = DataFile.load(dset='Train', net=net_type, run=run, epoch=epoch)
    validData = DataFile.load(dset='Valid', net=net_type, run=run, epoch=epoch)
    testData  = None

    print("Loaded {} entries to {} set".format(len(trainData), 'Train'))
    print("Loaded {} entries to {} set".format(len(validData), 'Valid'))
    print("Test data not available")

    def gather_data(data):
        return  [(  entry['embed'],
                    transform.resize(entry['mask'], output_shape=entry['embed'].shape, order=0),
                    entry['label'],
                    entry['info'],
                    entry['size'],
                    entry['rating'],
                    entry['weights'],
                    entry['z'])
                        for entry in data]

    validData = gather_data(validData)
    trainData = gather_data(trainData)

    image_ = np.concatenate([e[0].flatten() for e in trainData])
    print("\tImage Range = [{:.1f}, {:.1f}]".format(image_.max(), image_.min()))

    return testData, validData, trainData


def load_nodule_dataset(size=128, res=1.0, apply_mask_to_patch=False, sample='Normal', configuration=None, n_groups=5,
                        dataset_type='Clean'):
    if configuration is None:
        return load_nodule_dataset_old_style(size=size, res=res, apply_mask_to_patch=apply_mask_to_patch, sample=sample)

    if apply_mask_to_patch:
        print('WRN: apply_mask_to_patch is for debug only')


    test_id  = configuration
    valid_id = (configuration + 1) % n_groups
    testData, validData, trainData = [], [], []

    for c in range(n_groups):
        data_file = Dataset(data_type=dataset_type, conf=c, dir=input_dir)
        data_group = data_file.load(size, res, sample)

        if c == test_id:
            set = "Test"
            testData += data_group
        elif c == valid_id:
            set = "Valid"
            validData += data_group
        else:
            set = "Train"
            trainData += data_group
        print("Loaded {} entries from {} to {} set - LEGACY configuration".format(len(data_group), data_file.name(size, res, sample), set))

    #trainData, validData, testData = [reduce(lambda x, y: x + y, data) for data in [trainData, validData, testData]]

    def gather_data(data, apply_mask):
        return [(   entry['patch'] * (0.3 + 0.7 * entry['mask']) if apply_mask else entry['patch'],
                     entry['mask'],
                     entry['label'],
                     entry['info'],
                     entry['size'],
                     entry['rating'],
                     entry['weights'],
                     entry['z'])
                        for entry in data]

    testData = gather_data(testData, apply_mask_to_patch)
    validData = gather_data(validData, apply_mask_to_patch)
    trainData = gather_data(trainData, apply_mask_to_patch)

    image_ = np.concatenate([e[0].flatten() for e in trainData])
    print("\tImage Range = [{:.1f}, {:.1f}]".format(image_.max(), image_.min()))

    return testData, validData, trainData


def load_nodule_dataset_pred(size=128, res=1.0, apply_mask_to_patch=False, sample='Normal', configuration=None):
    if configuration is None:
        assert False

    if apply_mask_to_patch:
        print('WRN: apply_mask_to_patch is for debug only')

    dataset_type = 'Primary'

    manager = CrossValidationManager('PRED')
    print("Using {} CrossValidationManager:".format('PRED'))

    def load(data_type, conf, dir, size, res, sample, data_set):
        data_file = Dataset(data_type=data_type, conf=conf, dir=dir)
        data = data_file.load(size, res, sample)
        print("\tLoaded {} entries from {} to {} set".format(len(data), data_file.name(size, res, sample), data_set))
        return data

    trainData = [load(dataset_type, c, input_dir, size, res, sample, 'Train')
                 for c in manager.get_train(configuration)]

    validData = [load(dataset_type, c, input_dir, size, res, sample, 'Valid')
                 for c in manager.get_valid(configuration)]

    testData = [load(dataset_type, c, input_dir, size, res, sample, 'Test')
                for c in manager.get_test(configuration)]

    trainData, validData, testData = [reduce(lambda x, y: x + y, data) for data in [trainData, validData, testData]]

    def gather_data(data, apply_mask):
        return [(   entry['patch'] * (0.3 + 0.7 * entry['mask']) if apply_mask else entry['patch'],
                     entry['mask'],
                     entry['label'],
                     entry['info'],
                     entry['size'],
                     entry['rating'],
                     entry['weights'],
                     entry['z'])
                        for entry in data]

    testData = gather_data(testData, apply_mask_to_patch)
    validData = gather_data(validData, apply_mask_to_patch)
    trainData = gather_data(trainData, apply_mask_to_patch)

    image_ = np.concatenate([e[0].flatten() for e in trainData])
    print("\tImage Range = [{:.1f}, {:.1f}]".format(image_.max(), image_.min()))

    return testData, validData, trainData


def load_nodule_dataset_ret(run, epoch, apply_mask_to_patch=False, configuration=None):
    if configuration is None:
        assert False

    if apply_mask_to_patch:
        print('WRN: apply_mask_to_patch is for debug only')

    manager = CrossValidationManager('PRED')
    # it is used for RET runs, however the loaded data is based on PRED network
    print("Using {} CrossValidationManager:".format('RET'))

    def load(r, epoch, indir, data_set):
        data_file = DatasetFromPredication(type='rating', input_dir=indir)
        data = data_file.load(goal=data_set, run=r, epoch=epoch)
        print("\tLoaded {} entries from {} to {} set".format(len(data), data_file.name(goal=data_set, run=r), data_set))
        return data

    cnf_id = manager.get_run_id(configuration)
    run_name = '{}C{}'.format(run, cnf_id)

    trainData = load(run_name, epoch, output_dir, 'Train')

    validData = load(run_name, epoch, output_dir, 'Valid')

    testData = load(run_name, epoch, output_dir, 'Test')


    #trainData, validData, testData = [reduce(lambda x, y: x + y, data) for data in [trainData, validData, testData]]

    def gather_data(data, apply_mask):
        return [(   entry['patch'] * (0.3 + 0.7 * entry['mask']) if apply_mask else entry['patch'],
                     entry['mask'],
                     entry['label'],
                     entry['info'],
                     entry['size'],
                     entry['rating'],
                     entry['weights'],
                     entry['z'])
                        for entry in data]

    testData = gather_data(testData, apply_mask_to_patch)
    validData = gather_data(validData, apply_mask_to_patch)
    trainData = gather_data(trainData, apply_mask_to_patch)

    image_ = np.concatenate([e[0].flatten() for e in trainData])
    print("\tImage Range = [{:.1f}, {:.1f}]".format(image_.max(), image_.min()))

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
        filename = input_dir + '/Dataset{:.0f}-{}-{}.p'.format(size, res, sample)
    else:
        filename = input_dir + '/Dataset{:.0f}-{:.1f}-{}.p'.format(size, res, sample)

    print('Loading {}'.format(filename))

    try:
        testData, validData, trainData = pickle.load(open(filename, 'br'))
    except:
        print('...Failed')
        testData, validData, trainData = pickle.load(open('.'+filename, 'br'))

    return testData, validData, trainData

