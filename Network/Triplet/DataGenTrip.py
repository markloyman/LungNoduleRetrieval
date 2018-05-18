import numpy as np
from keras import utils

try:
    from Network.data_loader import load_nodule_dataset, prepare_data_triplet, prepare_data
    from Network.dataUtils import augment_all, crop_center, get_sample_weight_for_similarity, get_class_weight
except:
    from data_loader import load_nodule_dataset, prepare_data_triplet, prepare_data
    from dataUtils import augment_all, crop_center, get_sample_weight, get_class_weight

class DataGeneratorTrip(object):
    """docstring for DataGenerator"""

    def __init__(self,  data_size= 128, model_size=128, res='Legacy', sample='Normal', batch_size=32,
                        do_augment=False, augment=None, use_class_weight=False, class_weight='', use_confidence=False,
                        debug=False, val_factor=1, objective="malignancy", configuration=None):

        dataset = load_nodule_dataset(size=data_size, res=res, sample=sample, apply_mask_to_patch=debug,  configuration=configuration)

        self.train_set = dataset[2]
        self.valid_set = dataset[1]

        self.objective = objective
        self.batch_sz = batch_size
        self.data_size  = data_size
        self.model_size = model_size

        self.val_factor = val_factor

        self.use_class_weight = use_class_weight
        self.class_weight_method = class_weight

        self.do_augment = do_augment
        self.augment = augment
        if do_augment:
            assert(augment is not None)

        self.train_seq = DataSequenceTrip(self.train_set,
                                         model_size=model_size, batch_size=batch_size,
                                         is_training=True, objective=objective,
                                         do_augment=do_augment, augment=augment,
                                         use_class_weight=use_class_weight, use_confidence=use_confidence)

        if val_factor >= 1:
            self.val_seq = DataSequenceTrip(self.valid_set,
                                           model_size=model_size, batch_size=batch_size,
                                           is_training=False, objective=objective,
                                           do_augment=do_augment, augment=augment,
                                           use_class_weight=use_class_weight, use_confidence=use_confidence)
        else:
            self.val_seq = None

        self.test_set = None

        print("Trainings Sets: {}, Validation Sets: {}".format(len(self.train_seq),
                                                               len(self.val_seq) if self.val_seq is not None else 0))

    def training_sequence(self):
        return self.train_seq

    def validation_sequence(self):
        return self.val_seq

    def has_validation(self):
        return self.val_seq is not None

    def get_data(self, dataset, is_training):
        ret_conf = self.class_weight_method if self.use_class_weight else None
        data = prepare_data_triplet(set, verbose=self.verbose, objective=self.objective, return_confidence=ret_conf)
        return data

    def get_train_data(self):
        return self.get_data(self.train_set, is_training=True)

    def get_valid_data(self):
        return self.get_data(self.valid_set, is_training=False)

    def get_test_images(self):
        return self.get_data(self.test_set, is_training=False)

    def get_flat_data(self, dataset):
        objective = 'malignancy'
        images, labels, classes, masks, meta, conf = \
            prepare_data(dataset, objective=objective, categorize=(2 if (objective == 'malignancy') else 0),
                         verbose=True, reshuffle=False, return_meta=True)
        if self.model_size != self.data_size:
            images = np.array([crop_center(im, msk, size=self.model_size)[0]
                               for im, msk in zip(images, masks)])
        return images, labels, classes, masks, meta, conf

    def get_flat_train_data(self):
        return self.get_flat_data(self.train_set)

    def get_flat_valid_data(self):
        return self.get_flat_data(self.valid_set)

    def get_flat_test_images(self):
        return self.get_flat_data(self.test_set)


class DataSequenceTrip(utils.Sequence):
    def __init__(self, dataset, is_training=True, model_size=128, batch_size=32,
                 objective="malignancy", rating_scale='none',
                 do_augment=False, augment=None, use_class_weight=False, use_confidence=False, debug=False,
                 val_factor=1):
        assert (use_confidence is False)

        print('Run Gen: {}'.format(np.where(is_training, 'Training', 'Validation')))

        self.objective = objective
        self.rating_scale = rating_scale

        self.dataset = dataset
        self.is_training = is_training

        self.batch_size = batch_size
        self.model_size = model_size

        if is_training:
            self.N = 672 // self.batch_size  # len(self.dataset)
        else:
            self.N = val_factor * (len(self.dataset) // self.batch_size)

        if do_augment:
            assert(augment is not None)
        self.do_augment = do_augment
        self.augment = augment

        self.use_class_weight = use_class_weight
        self.class_weight_method = 'balanced'

        self.verbose = 1
        self.epoch = 0

        self.on_epoch_end()

    def on_epoch_end(self):

        ret_conf = self.class_weight_method if self.use_class_weight else None
        images, labels, masks, confidence = \
            prepare_data_triplet(self.dataset, verbose=self.verbose, objective=self.objective, return_confidence=ret_conf)

        if self.use_class_weight:
            assert False
            class_weight = get_class_weight(confidence, method=self.class_weight_method)
            sample_weight = get_sample_weight_for_similarity(confidence, wD=class_weight['D'], wSB=class_weight['SB'],
                                               wSM=class_weight['SM'])
            #if self.verbose == 1:
            #    print([(li, np.round(10 * wi, 2).astype('uint')) for li, wi in zip(lbl, w)])
        else:
            sample_weight = np.ones(labels.shape)

        # split into batches
        split_idx = [b for b in range(self.batch_size, images[0].shape[0], self.batch_size)]
        images = (np.array_split(images[0], split_idx),
                  np.array_split(images[1], split_idx),
                  np.array_split(images[2], split_idx))
        labels = np.array_split(labels, split_idx)
        masks = (np.array_split(masks[0], split_idx),
                 np.array_split(masks[1], split_idx),
                 np.array_split(masks[2], split_idx))
        sample_weight = np.array_split(sample_weight, split_idx)

        if self.verbose == 1:
            print("batch size:{}, sets:{}".format(images[0][0].shape[0], len(images[0])))

        # if last batch smaller than batch_sz, discard it
        if images[0][-1].shape[0] < self.batch_size:
            images = (images[0][:-1], images[1][:-1], images[2][:-1])
            labels = labels[:-1]
            masks = (masks[0][:-1], masks[1][:-1], masks[2][:-1])
            if self.use_class_weight:
                sample_weight = sample_weight[:-1]
            if self.verbose == 1:
                print("discard last unfull batch -> sets:{}".format(len(images[0])))

        assert self.N == images[0].shape[0]

        self.images = images
        self.labels = labels
        #self.classes = classes
        self.masks = masks
        self.sample_weight = sample_weight

        self.epoch = self.epoch + 1
        self.verbose = 0

    def __getitem__(self, index):
        print('idx = {}'.format(index))
        if self.do_augment and self.is_training and (self.epoch >= self.augment['epoch']):
            if index == 0:
                print("Begin augmenting")
            images_batch_0 = augment_all(self.images[0][index], self.masks[0][index],
                                       model_size=self.model_size, augment_params=self.augment)
            images_batch_1 = augment_all(self.images[1][index], self.masks[1][index],
                                       model_size=self.model_size, augment_params=self.augment)
            images_batch_2 = augment_all(self.images[2][index], self.masks[2][index],
                                         model_size=self.model_size, augment_params=self.augment)
        else:
            images_batch_0 = np.array([crop_center(im, msk, size=self.model_size)[0]
                               for im, msk in zip(self.images[0][index], self.masks[0][index])])
            images_batch_1 = np.array([crop_center(im, msk, size=self.model_size)[0]
                                       for im, msk in zip(self.images[1][index], self.masks[1][index])])
            images_batch_2 = np.array([crop_center(im, msk, size=self.model_size)[0]
                                       for im, msk in zip(self.images[2][index], self.masks[2][index])])
        labels_batch = self.labels[index]
        weights_batch = self.sample_weight[index]

        if index == 0:
            print("Batch #{} of size {}".format(index, images_batch_0.shape))

        return [images_batch_0, images_batch_1, images_batch_2], labels_batch, weights_batch

    def __len__(self):
        return self.N