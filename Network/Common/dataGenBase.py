import numpy as np
from keras import utils
try:
    from Network.data_loader import load_nodule_dataset, prepare_data, prepare_data_direct
    from Network.dataUtils import augment_all, crop_center_all, crop_center, get_sample_weight, get_class_weight
except:
    from data_loader import load_nodule_dataset, prepare_data, prepare_data_direct
    from dataUtils import augment_all, crop_center_all, crop_center, get_sample_weight, get_class_weight


def split(data_array, idx):
    if len(data_array) == 2:
        data_array = np.array_split(data_array[0], idx), np.array_split(data_array[1], idx)
    elif len(data_array) == 3:
            data_array = np.array_split(data_array[0], idx), np.array_split(data_array[1], idx), np.array_split(data_array[2], idx)
    else:
        data_array = np.array_split(data_array, idx)
    return data_array


class DataGeneratorBase(utils.Sequence):
    """docstring for DataGenerator"""

    def __init__(self,  data_size= 128, model_size=128, res='Legacy', sample='Normal', batch_size=32,
                        objective='malignancy', rating_scale='none', categorize=False,
                        do_augment=False, augment=None, weighted_rating=False,
                        use_class_weight=False, use_confidence=False,
                        val_factor = 0, balanced=False, configuration=None, train_factor=1,
                        debug=False, full=False, include_unknown=False):

        self.objective = objective
        self.rating_scale = rating_scale
        self.weighted_rating = weighted_rating

        dataset = load_nodule_dataset(size=data_size, res=res, sample=sample, configuration=configuration,
                                      full=full, include_unknown=include_unknown,
                                      apply_mask_to_patch=debug)

        self.test_set, self.valid_set, self.train_set = dataset

        self.batch_size = batch_size
        self.data_size  = data_size
        self.model_size = model_size
        self.balanced = balanced

        self.train_seq = self.get_sequence()(self.train_set,
                                             model_size=model_size, batch_size=batch_size, weighted_rating=weighted_rating,
                                             is_training=True, objective=objective, rating_scale=rating_scale,
                                             categorize=categorize, do_augment=do_augment, augment=augment,
                                             use_class_weight=use_class_weight, use_confidence=use_confidence,
                                             balanced=balanced, data_factor=train_factor)

        if val_factor > 1:
            self.val_seq = self.get_sequence()(self.valid_set,
                                               model_size=model_size, batch_size=batch_size, weighted_rating=weighted_rating,
                                               is_training=False, objective=objective, rating_scale=rating_scale,
                                               categorize=categorize, do_augment=do_augment, augment=augment,
                                               use_class_weight=use_class_weight, use_confidence=use_confidence,
                                               balanced=balanced)
        else:
            self.val_seq = None

        print("Trainings Sets: {}, Validation Sets: {}".format(len(self.train_seq), len(self.val_seq) if self.val_seq is not None else 0))

    def training_sequence(self):
        return self.train_seq

    def validation_sequence(self):
        return self.val_seq

    def has_validation(self):
        return self.val_seq is not None

    def get_train_data(self):
        return self.get_data(self.train_set, is_training=True)

    def get_valid_data(self):
        return self.get_data(self.valid_set, is_training=False)

    def get_test_images(self):
        return self.get_data(self.test_set, is_training=False)

    def get_flat_data(self, dataset):
        images, labels, classes, masks, meta, conf, nodule_size, _, z = \
            prepare_data(dataset, rating_format='raw', verbose=True, reshuffle=False, return_meta=True)
        if self.model_size != self.data_size:
            images = np.array([crop_center(im, msk, size=self.model_size)[0]
                               for im, msk in zip(images, masks)])
        return images, labels, classes, masks, meta, conf, nodule_size, z

    def get_flat_train_data(self):
        print('Loaded flat training data')
        return self.get_flat_data(self.train_set)

    def get_flat_valid_data(self):
        print('Loaded flat validation data')
        return self.get_flat_data(self.valid_set)

    def get_flat_test_data(self):
        print('Loaded flat test data')
        return self.get_flat_data(self.test_set)

    def get_data(self, dataset, is_training):
        raise NotImplementedError("get_data() is an abstract method")

    def get_sequence(self):
        raise NotImplementedError("get_sequences() is an abstract method")

    def set_regularizer(self, enable):
        self.train_seq.enable_regularization = enable


class DataSequenceBase(utils.Sequence):

    def __init__(self, dataset, is_training=True, model_size=128, batch_size=32,
                 objective='malignancy', rating_scale='none', categorize=False,
                 do_augment=False, augment=None, weighted_rating=False,
                 use_class_weight=False, use_confidence=False,
                 balanced=False, data_factor=1):

        self.objective = objective
        self.rating_scale = rating_scale
        self.weighted_rating = weighted_rating
        self.categorize = categorize
        self.dataset = dataset
        self.is_training = is_training
        self.batch_size = batch_size
        self.model_size = model_size
        self.balanced = balanced
        self.use_class_weight = use_class_weight
        self.use_confidence = use_confidence
        self.do_augment = do_augment
        self.augment = augment
        if do_augment:
            assert (augment is not None)
        self.verbose = 1
        self.epoch = 0

        if objective not in ['malignancy', 'rating', 'size', 'rating_size', 'distance-matrix']:
            print("ERR: Illegual objective given ({})".format(self.objective))
            assert False
        self.data_factor = data_factor
        self.N = self.calc_N(data_factor)
        print("DataSequence N = {}".format(self.N))

        self.enable_regularization = False

        self.on_epoch_end()

    def process_label_batch(self, labels_batch):
        return labels_batch

    def on_epoch_end(self):
        print('Run Gen {}: {}'.format(self.epoch, np.where(self.is_training, 'Training', 'Validation')))

        images, labels, classes, masks, sample_weights = zip(*[self.load_data() for i in range(self.data_factor)])
        assert len(images) == self.data_factor

        num_of_streams = len(images[0])
        images = [np.vstack([pair[i] for pair in images]) for i in range(num_of_streams)]
        masks  = [np.vstack([pair[i] for pair in masks])  for i in range(num_of_streams)]

        num_of_losses = len(labels[0])
        labels = [np.hstack([label[i] for label in labels]) for i in range(num_of_losses)]

        classes = np.hstack(classes)
        sample_weights = np.hstack(sample_weights)

        # split into batches
        num_of_images = images[0].shape[0]
        split_idx = [b for b in range(self.batch_size, num_of_images, self.batch_size)]
        images = tuple([np.array_split(image, split_idx) for image in images])
        #labels = np.array_split(labels, split_idx)
        labels = tuple([np.array_split(lbl, split_idx) for lbl in labels])
        classes = np.array_split(classes, split_idx)
        masks = tuple([np.array_split(mask, split_idx) for mask in masks])
        sample_weights = np.array_split(sample_weights, split_idx)

        batch_size = images[0][0].shape[0]
        number_of_batches = len(images[0])
        if self.verbose == 1:
            print("batch size:{}, sets:{}".format(batch_size, number_of_batches))
        assert batch_size == self.batch_size

        # if last batch smaller than batch_size, discard it
        last_batch = images[0][-1].shape[0]
        if last_batch < self.batch_size:
            images = tuple([im[:-1] for im in images])
            labels = tuple([lbl[:-1] for lbl in labels])
            classes = classes[:-1]
            masks = tuple([im[:-1] for im in masks])
            sample_weights = sample_weights[:-1]
            number_of_batches = len(images[0])
            if self.verbose == 1:
                print("discard last unfull batch -> sets:{}".format(number_of_batches))

        #if self.epoch == 0:
        #    assert number_of_batches == self.N  # len(self.dataset)//self.batch_size

        self.images = images
        self.labels = labels
        self.classes = classes
        self.masks = masks
        self.sample_weights = sample_weights

        self.epoch = self.epoch + 1
        self.verbose = 0

    def __getitem__(self, index):

        if self.do_augment and self.is_training and (self.epoch >= self.augment['epoch']):
            if index == 0:
                print("Augmentation Enabled")
            images_batch = [augment_all(images[index], masks[index], model_size=self.model_size, augment_params=self.augment)
                                for images, masks in zip(self.images, self.masks)]
        else:
            images_batch = [crop_center_all(images[index], masks[index], size=self.model_size)
                                for images, masks in zip(self.images, self.masks)]
        if self.weighted_rating:
            labels_batch = [self.process_label_batch(self.labels[0][index], self.labels[1][index])]
        else:
            labels_batch = [self.process_label_batch(lbl[index]) for lbl in self.labels]
        if self.enable_regularization:
            labels_batch.append(np.zeros(self.batch_size))

        # use same weights for all losses
        weights_batch = [self.sample_weights[index] for i in range(len(labels_batch))]

        if index == 0:
            batch_size = images_batch[0].shape if type(images_batch) is list else images_batch.shape
            print("Batch #{} of size {}".format(index, batch_size ))
            print("\tWeights: {}".format(weights_batch[:10]))

        return images_batch, \
               labels_batch if len(labels_batch) > 1 else labels_batch[0], \
               weights_batch if len(weights_batch)>1 else weights_batch[0]

    def __len__(self):
        return self.N

    def load_data(self):
        raise NotImplementedError("images, labels, masks, confidence = load_data() is an abstract method")

    def calc_N(self, data_factor):
        raise NotImplementedError("get_N() is an abstract method")
