import numpy as np
from keras import utils

try:
    from Network.data_loader import load_nodule_dataset, prepare_data, prepare_data_direct
    from Network.dataUtils import augment_all, crop_center, get_sample_weight, get_class_weight
except:
    from data_loader import load_nodule_dataset, prepare_data, prepare_data_direct
    from dataUtils import augment_all, crop_center, get_sample_weight, get_class_weight


def select_balanced(some_set, labels, N, permutation):
    b = some_set[0 == labels][:N]
    m = some_set[1 == labels][:N]
    merged = np.concatenate([b, m], axis=0)
    reshuff = merged[permutation]
    return reshuff


class DataGeneratorDir(object):
    """docstring for DataGenerator"""

    def __init__(self,  data_size= 128, model_size=128, res='Legacy', sample='Normal', batch_size=32,
                        objective='malignancy', rating_scale='none', categorize=False,
                        do_augment=False, augment=None,
                        use_class_weight=False, use_confidence=False,
                        val_factor = 0, balanced=False, configuration=None,
                        debug=False,):

        assert(categorize==False)
        assert(use_confidence==False)

        self.objective = objective
        self.rating_scale = rating_scale

        dataset = load_nodule_dataset(size=data_size, res=res, sample=sample, configuration=configuration,
                                      apply_mask_to_patch=debug)
        self.train_set = dataset[2]
        self.valid_set = dataset[1]

        self.batch_size = batch_size
        self.data_size  = data_size
        self.model_size = model_size

        self.train_seq = DataSequenceDir(self.train_set,
                                         model_size=model_size, batch_size=batch_size,
                                         is_training=True, objective=objective, rating_scale=rating_scale,
                                         categorize=categorize, do_augment=do_augment, augment=augment,
                                         use_class_weight=use_class_weight, use_confidence=use_confidence,
                                         balanced=balanced)

        if val_factor > 1:
            self.val_seq = DataSequenceDir(self.train_set,
                                             model_size=model_size, batch_size=batch_size,
                                             is_training=False, objective=objective, rating_scale=rating_scale,
                                             categorize=categorize, do_augment=do_augment, augment=augment,
                                             use_class_weight=use_class_weight, use_confidence=use_confidence,
                                             balanced=balanced)
        else:
            self.val_seq = None

        self.test_set = None

        print("Trainings Sets: {}, Validation Sets: {}".format(len(self.train_seq), len(self.val_seq) if self.val_seq is not None else 0))

    def training_sequence(self):
        return self.train_seq

    def validation_sequence(self):
        return self.val_seq

    def has_validation(self):
        return self.val_seq is not None

    def get_train_data(self):
        return prepare_data_direct(self.train_set, objective=self.objective, reshuffle=False,
                                   rating_scale=self.rating_scale, classes=2, size=self.model_size,
                                   verbose=True, return_meta=True)

    def get_valid_data(self):
        return prepare_data_direct(self.valid_set, objective=self.objective, reshuffle=False,
                                   rating_scale=self.rating_scale, classes=2, size=self.model_size,
                                   verbose=True, return_meta=True)

    def get_test_images(self):
        return prepare_data_direct(self.test_set, objective=self.objective, reshuffle=False,
                                   rating_scale=self.rating_scale, classes=2, size=self.model_size,
                                   verbose=True, return_meta=True)


class DataSequenceDir(utils.Sequence):

    def __init__(self, dataset, is_training=True, model_size=128, batch_size=32,
                 objective='malignancy', rating_scale='none', categorize=False,
                 do_augment=False, augment=None,
                 use_class_weight=False, use_confidence=False,
                 balanced=False, val_factor=1):

        assert (categorize is False)
        assert (use_confidence is False)

        self.objective = objective
        self.rating_scale = rating_scale

        self.dataset = dataset
        self.is_training = is_training

        self.batch_size = batch_size
        self.model_size = model_size

        if objective == 'malignancy':
            labels = np.array([entry[2] for entry in dataset])
            Nb = np.count_nonzero(1 - labels)
            Nm = np.count_nonzero(labels)

            if is_training:
                if balanced:
                    self.N = 2 * np.minimum(Nb, Nm) // self.batch_size
                    # self.trainN = 666 // self.batch_size
                else:
                    self.N = (Nb + Nm) // self.batch_size
                    # self.trainN = 1023 // self.batch_sz
            else:
                self.N = val_factor * (len(self.valid_set) // self.batch_size)  # 339

            self.balanced = balanced

            self.use_class_weight = use_class_weight
            # if use_class_weight:
            #    self.class_weight = get_class_weight(labels, class_weight)
            #    print("Class Weight -> Benign: {:.2f}, Malignant: {:.2f}".format(self.class_weight[0], self.class_weight[1]))
            # else:
            #    self.class_weight = None
        elif objective == 'rating':
            self.N = len(self.dataset) // batch_size
            if balanced:
                print("WRN: objective rating does not support balanced")
            self.balanced = False
            #if use_class_weight:
            #    print("WRN: objective rating does not support use class weight")
            #self.use_class_weight = False
            #self.class_weight = None
            self.use_class_weight = use_class_weight
        else:
            print("ERR: Illegual objective given ({})".format(objective))
            assert (False)

        self.do_augment = do_augment
        self.augment = augment
        if do_augment: assert (augment is not None)

        self.verbose = 1
        self.epoch = 0

        self.on_epoch_end()

    def on_epoch_end(self):
        print('Run Gen {}: {}'.format(self.epoch, np.where(self.is_training, 'Training', 'Validation')))
        #size = self.data_size if self.do_augment else self.model_size
        # images, labels, masks, confidence = \
        images, labels, classes, masks = \
            prepare_data_direct(self.dataset, objective=self.objective, rating_scale=self.rating_scale, classes=2,
                                size=self.model_size, verbose=self.verbose)[:4]

        if self.use_class_weight:
            class_weights = get_class_weight(np.squeeze(classes), 'balanced')
            print("Class Weight -> Benign: {:.2f}, Malignant: {:.2f}".format(class_weights[0], class_weights[1]))
            sample_weights = get_sample_weight(classes, class_weights)
        else:
            sample_weights = np.ones(len(labels))

        Nb = np.count_nonzero(1 - classes)
        Nm = np.count_nonzero(classes)
        N = np.minimum(Nb, Nm)
        if self.verbose:
            print("Benign: {}, Malignant: {}".format(Nb, Nm))
        if self.balanced and self.is_training:
            new_order = np.random.permutation(2 * N)
            labels_ = np.argmax(classes, axis=1)
            images = select_balanced(images, labels_, N, new_order)
            labels = select_balanced(labels, labels_, N, new_order)
            classes = select_balanced(classes, labels_, N, new_order)
            masks = select_balanced(masks, labels_, N, new_order)
            sample_weights = select_balanced(sample_weights, labels_, N, new_order)

            if self.verbose:
                Nb = np.count_nonzero(1 - np.argmax(classes, axis=1))
                Nm = np.count_nonzero(np.argmax(classes, axis=1))
                print("Balanced - Benign: {}, Malignant: {}".format(Nb, Nm))

        #if self.verbose:
        #    print("images after augment/crop: {}".format(images[0].shape))

        # split into batches
        split_idx = [b for b in range(self.batch_size, images.shape[0], self.batch_size)]
        images = np.array_split(images, split_idx)
        labels = np.array_split(labels, split_idx)
        classes = np.array_split(classes, split_idx)
        masks = np.array_split(masks, split_idx)
        sample_weights = np.array_split(sample_weights, split_idx)
        # confidence = np.array_split(confidence,  split_idx)

        if self.verbose == 1:
            print("batch size:{}, sets:{}".format(images[0].shape[0], len(images)))

        # if last batch smaller than batch_sz, discard it
        if images[-1].shape[0] < self.batch_size:
            images = images[:-1]
            labels = labels[:-1]
            classes = classes[:-1]
            masks = masks[:-1]
            sample_weights = sample_weights[:-1]
            # if self.use_class_weight:
            #    confidence = confidence[:-1]
            if self.verbose == 1:
                print("discard last unfull batch -> sets:{}".format(len(images)))

        if self.epoch == 0:
            assert (len(images) == len(self.dataset)//self.batch_size)

        self.images = images
        self.labels = labels
        self.classes = classes
        self.masks = masks
        self.sample_weights = sample_weights

        self.epoch = self.epoch + 1
        self.verbose = 0

    def __getitem__(self, index):

        if self.do_augment and self.is_training and (self.epoch >= self.augment['epoch']):
            #print("Begin augmenting")
            images_batch = augment_all(self.images[index], self.masks[index],
                                       model_size=self.model_size, augment_params=self.augment)
        else:
            images_batch = np.array([crop_center(im, msk, size=self.model_size)[0]
                               for im, msk in zip(self.images[index], self.masks[index])])
        labels_batch = self.labels[index]
        weights_batch = self.sample_weights[index]

        if index == 0:
            print("Batch #{} of size {}".format(index, images_batch.shape))

        return images_batch, labels_batch, weights_batch

    def __len__(self):
        return self.N