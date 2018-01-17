import numpy as np
import pickle
from Network.dataUtils import crop_center
from Network.data import load_nodule_dataset, prepare_data_direct
from Network.model import miniXception_loader
# from Network.metrics import siamese_margin
from Network.directArch import directArch


class Rating:
    def __init__(self, pooling='max'):
        self.pred_file_format = '.\output\embed\predR_dirR{}_E{}_{}.p'
        self.pooling = pooling

    def pred_filename(self, run, epoch, post):
        return self.pred_file_format.format(run, epoch, post)

    def load(self, run, epoch, post):
        filename = self.pred_filename(run=run, epoch=epoch, post=post)
        images, predict, meta_data, labels, masks = pickle.load(open(filename, 'br'))
        return images, predict, meta_data, labels, masks

    def predict_rating(self, weights_file, out_filename, data_subset_id, rating_scale='none'):

        # Setup
        size = 144
        input_shape = (128, 128, 1)
        sample = 'Normal'
        res = '0.5I'
        in_size = 128
        out_size = 128

        # prepare model
        model = directArch(miniXception_loader, input_shape, objective="rating", pooling=self.pooling, output_size=out_size,
                           normalize=True)
        if weights_file is not None:
            model.load_weights(weights_file)
            print('Load from: {}'.format(weights_file))
        else:
            print('Model without weights')

        # prepare test data
        images_test, labels_test, classes_test, masks_test, meta_test = \
            prepare_data_direct(load_nodule_dataset(size=size, res=res, sample=sample)[data_subset_id], size=size,
                                return_meta=True, objective="rating", rating_scale=rating_scale, verbose=1, balanced=False)

        print("Data ready: images({}), labels({})".format(images_test[0].shape, labels_test.shape))
        print("Range = [{:.2f},{:.2f}]".format(np.min(images_test[0]), np.max(images_test[0])))

        images_test = np.array([crop_center(im, msk, size=in_size)[0]
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



