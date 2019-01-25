import numpy as np
from keras import utils
from functools import reduce
try:
    from Network.data_loader import load_nodule_dataset
    from Network.Common import prepare_data
    from Network.dataUtils import augment_all, crop_center_all, crop_center, get_sample_weight, get_class_weight, format_data_as_sequence
except:
    from data_loader import load_nodule_dataset
    from Common import prepare_data
    from dataUtils import augment_all, crop_center_all, crop_center, get_sample_weight, get_class_weight, format_data_as_sequence


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

    def __init__(self,  load_dataset_fn, data_size= 128, model_size=128, batch_size=32,
                        objective='malignancy', rating_scale='none', categorize=False,
                        do_augment=False, augment=None, weighted_rating=False,
                        use_class_weight=False, use_confidence=False,
                        val_factor = 0, balanced=False, train_factor=1,
                        seq_model=False):

        self.seq_model = seq_model
        self.objective = objective
        self.rating_scale = rating_scale
        self.weighted_rating = weighted_rating

        self.test_set, self.valid_set, self.train_set = load_dataset_fn()

        self.batch_size = batch_size
        self.data_size  = data_size
        self.model_size = model_size
        self.balanced = balanced

        # for sequence definition
        self.train_seq = None
        self.val_seq = None
        self.categorize = categorize
        self.do_augment = do_augment
        self.augment = augment
        self.use_class_weight = use_class_weight
        self.use_confidence = use_confidence
        self.val_factor = val_factor
        self.train_factor = train_factor

    def activate_sequences(self):
        self.train_seq = self.get_sequence()(self.train_set,
                                             model_size=self.model_size, batch_size=self.batch_size,
                                             weighted_rating=self.weighted_rating, rating_scale=self.rating_scale,
                                             objective=self.objective, seq_model=self.seq_model,
                                             is_training=True, data_factor=self.train_factor,
                                             balanced=self.balanced, categorize=self.categorize,
                                             do_augment=self.do_augment, augment=self.augment,
                                             use_class_weight=self.use_class_weight, use_confidence=self.use_confidence)

        if self.val_factor > 0:
            self.val_seq = self.get_sequence()(self.valid_set,
                                               model_size=self.model_size, batch_size=self.batch_size,
                                               weighted_rating=self.weighted_rating, rating_scale=self.rating_scale,
                                               objective=self.objective, seq_model=self.seq_model,
                                               is_training=False, data_factor=self.val_factor,
                                               balanced=self.balanced, categorize=self.categorize,
                                               do_augment=self.do_augment, augment=self.augment,
                                               use_class_weight=self.use_class_weight,
                                               use_confidence=self.use_confidence)

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

    def get_flat_data(self, dataset, resize=True):
        images, labels, classes, masks, meta, conf, nodule_size, rating_weights, z = \
            prepare_data(dataset, rating_format='raw', verbose=True, reshuffle=False)

        if self.model_size != self.data_size:
            if self.seq_model:
                images = format_data_as_sequence(images, embed_size=self.model_size)
            else:
                if resize:
                    images = np.array([crop_center(im, msk, size=self.model_size)[0]
                                   for im, msk in zip(images, masks)])

        return images, labels, classes, masks, meta, conf, nodule_size, rating_weights, z

    def get_flat_train_data(self, resize=True):
        print('Loaded flat training data')
        return self.get_flat_data(self.train_set, resize)

    def get_flat_valid_data(self, resize=True):
        print('Loaded flat validation data')
        return self.get_flat_data(self.valid_set, resize)

    def get_flat_test_data(self, resize=True):
        print('Loaded flat test data')
        return self.get_flat_data(self.test_set, resize)

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
                 balanced=False, data_factor=1, seq_model=False):

        self.seq_model = seq_model
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
        # heirarchy: images[data_factor][data_stream][dataset-entry]
        num_of_streams = len(images[0])

        def combine(x):
            if type(x[0]) is list:
                return reduce(lambda a, b: a + b, x)
            elif type(x[0]) is np.ndarray:
                return np.vstack(x)
            else:
                assert False

        images = [combine([pair[i] for pair in images]) for i in range(num_of_streams)]
        masks  = [combine([pair[i] for pair in masks])  for i in range(num_of_streams)]

        num_of_losses = len(labels[0])
        # triplet: vstack
        # num_of_losses > 1: hstack
        labels = [np.hstack([label[i] for label in labels]) for i in range(num_of_losses)]

        classes = np.hstack(classes)
        sample_weights = np.hstack(sample_weights)

        # split into batches
        def split(x, split_idx, batch):
            if type(x) is list:
                X = [x[(idx-batch):idx] for idx in split_idx]
                X += [x[split_idx[-1]:]]
                return X
            elif type(x) is np.ndarray:
                return np.array_split(x, split_idx)
            else:
                assert False

        num_of_images = len(images[0])
        split_idx = [b for b in range(self.batch_size, num_of_images, self.batch_size)]
        images = tuple([split(image, split_idx, self.batch_size) for image in images])
        #labels = np.array_split(labels, split_idx)
        labels = tuple([np.array_split(lbl, split_idx) for lbl in labels])
        classes = np.array_split(classes, split_idx)
        masks = tuple([split(mask, split_idx, self.batch_size) for mask in masks])
        sample_weights = np.array_split(sample_weights, split_idx)

        batch_size = len(images[0][0])
        number_of_batches = len(images[0])
        if self.verbose == 1:
            print("batch size:{}, sets:{}".format(batch_size, number_of_batches))
        assert batch_size == self.batch_size

        # if last batch smaller than batch_size, discard it
        last_batch = len(images[0][-1])
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
            #if index == 0:
            #    print("Augmentation Enabled")
            images_batch = [augment_all(images[index], masks[index], model_size=self.model_size, augment_params=self.augment)
                                for images, masks in zip(self.images, self.masks)]
        else:
            if self.seq_model:
                if len(self.images) == 1:
                    images_batch = self.images[0][index]
                else:
                    images_batch = [images[index] for images in zip(self.images)]
            else:
                images_batch = [crop_center_all(images[index], masks[index], size=self.model_size)
                                    for images, masks in zip(self.images, self.masks)]

        if self.weighted_rating and self.objective == 'distance-matrix':
            labels_batch = [self.process_label_batch(self.labels[0][index], self.labels[1][index])]
        else:
            labels_batch = [self.process_label_batch(lbl[index]) for lbl in self.labels]

        if self.enable_regularization:
            labels_batch.append(np.zeros(self.batch_size))

        # use same weights for all losses
        weights_batch = [self.sample_weights[index] for i in range(len(labels_batch))]

        # sequence padding (each batch must have constant sequence length
        if self.seq_model:
            #images_batch = np.array([im.swapaxes(0, 2).reshape([im.shape[2], 8*8*128]) for im in images_batch])
            images_batch = format_data_as_sequence(images_batch, embed_size=self.model_size)
            seq_len = np.array([im.shape[0] for im in images_batch])

            images_batch = np.array([np.vstack([np.zeros((seq_len.max() - im.shape[0], im.shape[1])),
                                                im])
                            for im in images_batch])
            #weights_batch = [np.vstack([np.repeat(np.expand_dims(w, axis=0), s, axis=0),
            #                            np.zeros((seq_len.max() - s, w.shape[0]))])
            #                 for w, s in zip(weights_batch, seq_len)]

        if index == 0:
            batch_size = images_batch[0].shape if type(images_batch) is list else images_batch.shape
            print("Batch #{} of size {}".format(index, batch_size ))
            # print("\tWeights: {}".format(weights_batch[:10]))

        return images_batch, \
               labels_batch if len(labels_batch) > 1 else labels_batch[0], \
               weights_batch if len(weights_batch)>1 else weights_batch[0]

    def __len__(self):
        return self.N

    def load_data(self):
        raise NotImplementedError("images, labels, masks, confidence = load_data() is an abstract method")

    def calc_N(self, data_factor):
        raise NotImplementedError("get_N() is an abstract method")
