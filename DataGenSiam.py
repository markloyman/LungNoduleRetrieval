import numpy as np
np.random.seed(1987)  # for reproducibility
import random

from keras import backend as K

from data import prepare_data, load_nodule_dataset, prepare_data_siamese, prepare_data_siamese_overlap, prepare_data_siamese_chained

class DataGenerator(object):
    """docstring for DataGenerator"""

    def __init__(self, im_size = 128, batch_sz=32, method = 'base'):

        dataset = load_nodule_dataset()

        self.train_set = dataset[2]
        self.valid_set = dataset[1]

        self.batch_sz = batch_sz
        self.im_size  = im_size

        if method is 'chained':
            self.trainN = 1670 // self.batch_sz
            self.valN   = 555  // self.batch_sz
        else:
            self.trainN = len(self.train_set)//self.batch_sz
            self.valN   = len(self.valid_set)//self.batch_sz

        self.method = method

        print("Method: {}, Trainings Sets: {}, Validation Sets: {}".format(method, self.trainN, self.valN))

    def next(self, set):
        verbose = 1
        while 1:
            if self.method is 'base':
                images_train, labels_train = prepare_data_siamese(set, size=self.im_size, verbose=verbose)
            elif self.method is 'overlapped':
                images_train, labels_train = prepare_data_siamese_overlap(set, size=self.im_size, verbose=verbose)
            elif self.method is 'chained':
                images_train, labels_train = prepare_data_siamese_chained(set, size=self.im_size, verbose=verbose)
            else:
                assert False
            if verbose == 0: print("reload data ({})".format(images_train[0].shape[0]))

            # split into batches
            split_idx = [b for b in range(self.batch_sz, images_train[0].shape[0], self.batch_sz)]
            images_train = ( np.array_split(images_train[0], split_idx),
                             np.array_split(images_train[1], split_idx) )
            labels_train = np.array_split(labels_train,  split_idx)

            if verbose == 1: print("batch size:{}, sets:{}".format(images_train[0][0].shape[0], len(images_train[0])))

            # if last batch smaller than batch_sz, discard it
            if images_train[0][-1].shape[0] < self.batch_sz:
                images_train = (images_train[0][:-1], images_train[1][:-1])
                labels_train = labels_train[:-1]
                if verbose == 1: print(
                print("discard last unfull batch -> sets:{}".format(len(images_train[0]))))

            verbose = 0
            for im0, im1, lbl in zip(images_train[0], images_train[1], labels_train):
                yield ([im0,im1],lbl)

    def next_train(self):
        return self.next(self.train_set)

    def next_val(self):
        return self.next(self.valid_set)

    def train_N(self):
        return self.trainN

    def val_N(self):
        return self.valN