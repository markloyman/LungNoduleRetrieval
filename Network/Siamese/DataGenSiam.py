import numpy as np

try:
    from Network.data import load_nodule_dataset, prepare_data_siamese, prepare_data_siamese_simple
    from Network.dataUtils import augment, crop_center, get_sample_weight, get_class_weight
    from Network.Siamese.metrics import siamese_rating_factor
except:
    from data import load_nodule_dataset, prepare_data_siamese, prepare_data_siamese_simple
    from dataUtils import augment, crop_center, get_sample_weight, get_class_weight
    from Siamese.metrics import siamese_rating_factor

class DataGenerator(object):
    """docstring for DataGenerator"""

    def __init__(self,  data_size= 128, model_size=128, res='Legacy', sample='Normal', batch_sz=32, objective="malignancy",
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
        if objective == "malignancy":
            if balanced:
                self.trainN = 1332 // self.batch_sz
            else:
                self.trainN = 1689 // self.batch_sz
            self.valN = val_factor * (559 // self.batch_sz)
            self.balanced = balanced
            self.use_class_weight = use_class_weight
        elif objective == "rating":
            self.trainN = len(self.train_set) // self.batch_sz
            self.valN = val_factor * (len(self.valid_set) // self.batch_sz)
            self.balanced = False
            self.use_class_weight = False

        self.class_weight_method = class_weight
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

    def next(self, set, is_training=False):
        verbose = 1
        epoch = 0
        while 1:
            print('Run Gen: {}'.format(np.where(is_training, 'Training', 'Validation')))
            size = self.data_size if self.do_augment else self.model_size
            if self.objective == "malignancy":
                images, labels, masks, confidence = \
                    prepare_data_siamese(set, size=size, balanced=(self.balanced and is_training),
                                         objective=self.objective, verbose=verbose)
            elif self.objective == "rating":
                images, labels, masks, confidence = \
                    prepare_data_siamese_simple(set, size=size, balanced=(self.balanced and is_training),
                                         objective=self.objective, verbose=verbose)
                labels *= siamese_rating_factor
                #images = np.repeat(images[0], R, axis=0), np.repeat(images[1], R, axis=0)
                #labels = np.repeat(labels, R, axis=0)
                #masks = np.repeat(masks[0], R, axis=0), np.repeat(masks[1], R, axis=0)
                #confidence = np.repeat(confidence, R, axis=0)
            else:
                assert(False)

            if self.do_augment and is_training and (epoch >= self.augment['epoch']):
                    if epoch == self.augment['epoch']:
                        print("Begin augmenting")
                    images = (self.augment_all(images[0], masks[0]), self.augment_all(images[1], masks[1]))
            else:
                images = (np.array([crop_center(im, msk, size=self.model_size)[0]
                                    for im, msk in zip(images[0], masks[0])]),
                          np.array([crop_center(im, msk, size=self.model_size)[0]
                                    for im, msk in zip(images[1], masks[1])]))

            if verbose:
                print("images after augment/crop: {}".format(images[0].shape))

            if self.use_class_weight:
                class_weight = get_class_weight(confidence, method=self.class_weight_method)

            # split into batches
            split_idx = [b for b in range(self.batch_sz, images[0].shape[0], self.batch_sz)]
            images = (  np.array_split(images[0], split_idx),
                        np.array_split(images[1], split_idx) )
            labels = np.array_split(labels,  split_idx)
            masks  = (  np.array_split(masks[0], split_idx),
                        np.array_split(masks[1], split_idx) )
            confidence = np.array_split(confidence,  split_idx)

            if verbose == 1: print("batch size:{}, sets:{}".format(images[0][0].shape[0], len(images[0])))

            # if last batch smaller than batch_sz, discard it
            if images[0][-1].shape[0] < self.batch_sz:
                images = (images[0][:-1], images[1][:-1])
                labels = labels[:-1]
                masks  = (masks[0][:-1], masks[1][:-1])
                if self.use_class_weight:
                    confidence = confidence[:-1]
                if verbose == 1:
                    print("discard last unfull batch -> sets:{}".format(len(images[0])))

            #if is_training:
            #    assert(len(images[0]) == self.trainN)
            #else:
            #    assert( (self.val_factor*len(images[0])) == self.valN)

            for im0, im1, lbl, msk0, msk1, cnf in zip(images[0], images[1], labels, masks[0], masks[1], confidence):
                if self.use_class_weight:
                    w = get_sample_weight(cnf,  wD=class_weight['D'],
                                                wSB=class_weight['SB'],
                                                wSM=class_weight['SM']
                                          )
                    if verbose == 1:
                        print([(li, np.round(10*wi, 2).astype('uint')) for li, wi in zip(lbl, w)])
                    verbose = 0
                    yield ([im0, im1], lbl, w)
                else:
                    yield ([im0, im1], lbl)
            epoch = epoch +1
            verbose = 0


    def next_train(self):
        return self.next(self.train_set, is_training=True)

    def next_val(self):
        return self.next(self.valid_set, is_training=False)

    def train_N(self):
        #return 5
        return self.trainN

    def val_N(self):
        #return 2
        return self.valN