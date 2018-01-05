import numpy as np

try:
    from Network.data import load_nodule_dataset, prepare_data, prepare_data_direct
    from Network.dataUtils import augment, crop_center, get_sample_weight, get_class_weight
except:
    from data import load_nodule_dataset, prepare_data, prepare_data_direct
    from dataUtils import augment, crop_center, get_sample_weight, get_class_weight

class DataGeneratorDir(object):
    """docstring for DataGenerator"""

    def __init__(self,  data_size= 128, model_size=128, res='Legacy', sample='Normal', batch_sz=32, objective='malignancy',
                        do_augment=False, augment=None, use_class_weight=False, class_weight='dummy', debug=False,
                        val_factor = 1, balanced=False):

        self.objective = objective

        dataset = load_nodule_dataset(size=data_size, res=res, sample=sample, apply_mask_to_patch=debug)
        self.train_set = dataset[2]
        self.valid_set = dataset[1]

        self.batch_sz = batch_sz
        self.data_size  = data_size
        self.model_size = model_size

        self.val_factor = val_factor

        if objective == 'malignancy':
            labels = np.array([entry[2] for entry in dataset[2]])
            Nb = np.count_nonzero(1 - labels)
            Nm = np.count_nonzero(labels)

            if balanced:
                self.trainN = 2*np.minimum(Nb, Nm) // self.batch_sz
                #self.trainN = 666 // self.batch_sz
            else:
                self.trainN = (Nb+Nm) // self.batch_sz
                #self.trainN = 1023 // self.batch_sz
            self.valN = val_factor * (339 // self.batch_sz)

            self.balanced = balanced

            self.use_class_weight = use_class_weight
            if use_class_weight:
                self.class_weight = get_class_weight(labels, class_weight)
                print("Class Weight -> Benign: {:.2f}, Malignant: {:.2f}".format(self.class_weight[0], self.class_weight[1]))
            else:
                self.class_weight = None
        elif objective == 'rating':
            self.trainN = len(self.train_set) // batch_sz
            self.valN = len(self.valid_set) // batch_sz
            if balanced:
                print("WRN: objective rating does not support balanced")
            self.balanced = False
            if use_class_weight:
                print("WRN: objective rating does not support use class weight")
            self.use_class_weight = False
            self.class_weight = None
        else:
            print("ERR: Illegual objective given ({})".format(objective))
            assert (False)

        self.do_augment = do_augment
        self.augment = augment
        if do_augment: assert(augment is not None)

        print("Trainings Sets: {}, Validation Sets: {}".format(self.trainN, self.valN))

    def augment_all(self, images, masks):
        images = [augment(im, msk,
                          size=self.model_size,
                          max_angle=self.augment['max_angle'],
                          flip_ratio=self.augment['flip_ratio'],
                          crop_stdev=self.augment['crop_stdev'])[0]
                  for im, msk in zip(images, masks)]
        return np.array(images)

    def select_balanced(self, some_set, labels, N, permutation):
        b = some_set[0 == labels][:N]
        m = some_set[1 == labels][:N]
        merged = np.concatenate([b, m], axis=0)
        reshuff = merged[permutation]
        return reshuff

    def next(self, set, is_training=False):
        verbose = 1
        epoch = 0
        while 1:
            print('Run Gen {}: {}'.format(epoch, np.where(is_training, 'Training', 'Validation')))
            size = self.data_size if self.do_augment else self.model_size
            #images, labels, masks, confidence = \
            images, labels, classes, masks = \
                prepare_data_direct(set, objective=self.objective, classes=2, size=self.model_size, verbose=verbose)
            #prepare_data(set, classes=2, verbose=verbose, reshuffle=True)
            Nb = np.count_nonzero(1-classes)
            Nm = np.count_nonzero(classes)
            N = np.minimum(Nb, Nm)
            if verbose:
                print("Benign: {}, Malignant: {}".format(Nb, Nm))
            if self.balanced and is_training:
                new_order = np.random.permutation(2*N)
                labels_ = np.argmax(classes, axis=1)
                images = self.select_balanced(images, labels_, N, new_order)
                labels = self.select_balanced(labels, labels_, N, new_order)
                classes = self.select_balanced(classes, labels_, N, new_order)
                masks = self.select_balanced(masks, labels_, N, new_order)
                if verbose:
                    Nb = np.count_nonzero(1 - np.argmax(classes, axis=1))
                    Nm = np.count_nonzero(np.argmax(classes, axis=1))
                    print("Balanced - Benign: {}, Malignant: {}".format(Nb, Nm))
            if self.do_augment and is_training and (epoch >= self.augment['epoch']):
                    if epoch == self.augment['epoch']:
                        print("Begin augmenting")
                    images = self.augment_all(images, masks)
            else:
                images = np.array([crop_center(im, msk, size=self.model_size)[0]
                                    for im, msk in zip(images, masks)])
            if verbose:
                print("images after augment/crop: {}".format(images[0].shape))

            #if self.use_class_weight:
            #    class_weight = get_class_weight(confidence, method=self.class_weight_method)

            # split into batches
            split_idx = [b for b in range(self.batch_sz, images.shape[0], self.batch_sz)]
            images = np.array_split(images, split_idx)
            labels = np.array_split(labels,  split_idx)
            classes = np.array_split(classes, split_idx)
            masks  = np.array_split(masks, split_idx)
            #confidence = np.array_split(confidence,  split_idx)

            if verbose == 1: print("batch size:{}, sets:{}".format(images[0].shape[0], len(images[0])))

            # if last batch smaller than batch_sz, discard it
            if images[-1].shape[0] < self.batch_sz:
                images = images[:-1]
                labels = labels[:-1]
                classes = classes[:-1]
                masks  = masks[:-1]
                #if self.use_class_weight:
                #    confidence = confidence[:-1]
                if verbose == 1:
                    print("discard last unfull batch -> sets:{}".format(len(images)))

            if epoch == 0:
                if is_training:
                    assert(len(images) == self.trainN)
                else:
                    assert( (self.val_factor*len(images)) == self.valN)

            #for im, lbl, msk, cnf in zip(images, labels, masks, confidence):
            for im, lbl, msk in zip(images, labels, masks):
                yield (im, lbl)
                #if self.use_class_weight:
                #    assert(False)
                #    w = get_sample_weight(cnf,  wD=class_weight['D'],
                #                                wSB=class_weight['SB'],
                #                                wSM=class_weight['SM']
                #                          )
                #    if verbose == 1:
                #        print([(li, np.round(10*wi, 2).astype('uint')) for li, wi in zip(lbl, w)])
                #    verbose = 0
                #    yield (im, lbl, w)
                #else:
                #    yield (im, lbl)
            epoch = epoch + 1
            verbose = 0

    def next_train(self):
        return self.next(self.train_set, is_training=True)

    def next_val(self):
        return self.next(self.valid_set, is_training=False)

    def train_N(self):
        return self.trainN

    def val_N(self):
        return self.valN