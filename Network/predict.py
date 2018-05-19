import pickle
import numpy as np

# from Network.metrics import siamese_margin
from Network.Direct.directArch import DirectArch
from Network.data_loader import load_nodule_dataset, prepare_data_direct
from Network.dataUtils import crop_center
from Network.model import miniXception_loader


class Rating:
    def __init__(self, pooling='max'):
        self.pred_file_format = '.\output\embed\predR_dirR{}_E{}_{}.p'
        self.pooling = pooling

        # Setup
        self.in_size = 128
        self.out_size = 128
        input_shape = (self.in_size, self.in_size, 1)

        # prepare model
        self.model = DirectArch(miniXception_loader, input_shape, objective="rating", pooling=self.pooling, output_size=self.out_size,
                           normalize=True)

    def load_dataset(self, data_subset_id, size=160, sample='Normal', res=0.5, rating_scale='none', configuration=None):
        # prepare test data
        self.images_test, self.labels_test, self.classes_test, self.masks_test, self.meta_test = \
            prepare_data_direct(load_nodule_dataset(configuration=configuration, size=size, res=res, sample=sample)[data_subset_id],
                                size=size,
                                return_meta=True, objective="rating", rating_scale=rating_scale, verbose=1,
                                balanced=False)

        print("Data ready: images({}), labels({})".format(self.images_test[0].shape, self.labels_test.shape))
        print("Range = [{:.2f},{:.2f}]".format(np.min(self.images_test[0]), np.max(self.images_test[0])))

        self.images_test = np.array([crop_center(im, msk, size=self.in_size)[0]
                                for im, msk in zip(self.images_test, self.masks_test)])

        print("Image size changed to {}".format(self.images_test.shape))
        print('Mask not updated')

    def pred_filename(self, run, epoch, post):
        return self.pred_file_format.format(run, epoch, post)

    def load(self, run, epoch, post):
        filename = self.pred_filename(run=run, epoch=epoch, post=post)
        predict, epochs, images, meta_data, labels, masks = pickle.load(open(filename, 'br'))
        return predict, epochs, images, meta_data, labels, masks

    def predict_rating(self, weights_file, out_filename):

        if weights_file is not None:
            self.model.load_weights(weights_file)
            print('Load from: {}'.format(weights_file))
        else:
            print('Model without weights')

        # eval
        print("Begin Predicting...")
        pred = self.model.predict(self.images_test, round=False)
        print("Predication Ready")
        print("\tshape = {}\n\trange [{}, {}]".format(pred.shape, np.min(pred), np.max(pred)))


        #pickle.dump((images_test, pred, meta_test, labels_test, masks_test), open(out_filename, 'bw'))
        #print("Saved to: {}".format(out_filename))

        return (self.images_test, pred, self.meta_test, self.classes_test, self.labels_test, self.masks_test), out_filename



class Malignancy:
    def __init__(self, pooling='max'):
        self.pred_file_format = '.\output\embed\pred_dir{}_E{}_{}.p'
        self.pooling = pooling
        # Setup
        self.data_size = 144
        self.sample = 'Normal'
        self.res = '0.5I'
        self.model_size = 128
        self.out_size = 128

    def pred_filename(self, run, epoch, post):
        return self.pred_file_format.format(run, epoch, post)

    def load(self, run, epoch, post):
        filename = self.pred_filename(run=run, epoch=epoch, post=post)
        images, predict, meta_data, labels, masks = pickle.load(open(filename, 'br'))
        return images, predict, meta_data, labels, masks

    def predict_malignancy(self, weights_file, out_filename, data_subset_id, configuration=None):

        input_shape = (self.model_size, self.model_size, 1)

        # prepare model
        model = DirectArch(miniXception_loader, input_shape, objective="malignancy", pooling=self.pooling, output_size=self.out_size,
                           normalize=True)
        if weights_file is not None:
            model.load_weights(weights_file)
            print('Load from: {}'.format(weights_file))
        else:
            print('Model without weights')

        # prepare test data
        images_test, labels_test, classes_test, masks_test, meta_test = \
            prepare_data_direct(
                load_nodule_dataset(configuration=configuration, size=self.data_size, res=self.res, sample=self.sample)[data_subset_id],
                    size=self.model_size, return_meta=True, objective="malignancy", verbose=1, balanced=False)

        print("Data ready: images({}), labels({})".format(images_test[0].shape, labels_test.shape))
        print("Range = [{:.2f},{:.2f}]".format(np.min(images_test[0]), np.max(images_test[0])))

        images_test = np.array([crop_center(im, msk, size=self.model_size)[0]
                                 for im, msk in zip(images_test, masks_test)])

        print("Image size changed to {}".format(images_test.shape))
        print('Mask not updated')

        # eval
        print("Begin Predicting...")
        pred = model.predict(images_test, round=False)
        print("Predication Ready")
        print("\tshape = {}\n\trange [{}, {}]".format(pred.shape, np.min(pred), np.max(pred)))


        pickle.dump((images_test, pred, meta_test, labels_test, masks_test), open(out_filename, 'bw'))
        print("Saved to: {}".format(out_filename))

        return (images_test, pred, meta_test, labels_test, masks_test), out_filename
