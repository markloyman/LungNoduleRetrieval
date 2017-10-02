import numpy as np
#np.random.seed(1987)  # for reproducibility

from Network.data import load_nodule_dataset, prepare_data_siamese
from Network.dataUtils import augment, get_sample_weight

class DataGenerator(object):
    """docstring for DataGenerator"""

    def __init__(self, data_size= 128, model_size=128, res='Legacy', batch_sz=32, do_augment=False, use_class_weight=False, debug=False):

        dataset = load_nodule_dataset(size=data_size, res=res, apply_mask_to_patch=debug)

        self.train_set = dataset[2]
        self.valid_set = dataset[1]

        self.batch_sz = batch_sz
        self.data_size  = data_size
        self.model_size = model_size

        self.trainN = 1670 // self.batch_sz
        self.valN   = 555  // self.batch_sz

        self.use_class_weight = use_class_weight
        self.do_augment       = do_augment

        print("Trainings Sets: {}, Validation Sets: {}".format(self.trainN, self.valN))

    def next(self, set):
        verbose = 1
        while 1:
            size = self.data_size if self.do_augment else self.model_size
            images, labels, masks, confidence = \
                prepare_data_siamese(set, size=size, verbose=verbose)
            if verbose == 0: print("reload data ({})".format(images[0].shape[0]))

            if self.do_augment:
                images = (np.array([augment(im, msk, size=self.model_size, max_angle=10, flip_ratio=0.1)[0]
                                    for im, msk in zip(images[0], masks[0])]),
                          np.array([augment(im, msk, size=self.model_size, max_angle=10, flip_ratio=0.1)[0]
                                    for im, msk in zip(images[1], masks[1])]))
                print("images after augment: {}".format(images[0].shape))

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

            verbose = 0
            for im0, im1, lbl, msk0, msk1, cnf in zip(images[0], images[1], labels, masks[0], masks[1], confidence):
                if self.use_class_weight:
                    w = get_sample_weight(cnf)
                    yield ([im0, im1], lbl, w)
                else:
                    yield ([im0, im1], lbl)


    def next_train(self):
        return self.next(self.train_set)

    def next_val(self):
        return self.next(self.valid_set)

    def train_N(self):
        return self.trainN

    def val_N(self):
        return self.valN