import pickle

import numpy as np

import FileManager
from Network.Direct.directArch import directArch
from Network.Siamese.siameseArch import siamArch
from Network.Triplet.tripletArch import tripArch
from Network.data_loader import load_nodule_dataset, prepare_data
from Network.dataUtils import crop_center
from Network.model import miniXception_loader


class Embeder:

    def __init__(self, network = 'dir', pooling='max', categorize=False):
        self.network = network
        self.Weights = FileManager.Weights(network)
        self.Embed = FileManager.Embed(network)

        self.data_size = 144
        self.data_res = '0.5I'  # 'Legacy'
        self.data_sample = 'Normal'

        self.net_in_size = 128
        self.net_input_shape = (self.net_in_size, self.net_in_size, 1)
        self.net_out_size = 128
        self.net_normalize = True
        self.net_pool = pooling
        self.categorize = categorize

    def prepare_data(self, data_subset_id):
        images, labels, classes, masks, meta, conf = \
            prepare_data(load_nodule_dataset(size=self.data_size, res=self.data_res, sample=self.data_sample)[data_subset_id],
                         categorize=False,
                         reshuffle=False,
                         return_meta=True,
                         verbose=1)
        self.images = np.array([crop_center(im, msk, size=self.net_in_size)[0]
                           for im, msk in zip(images, masks)])
        self.meta   = meta
        self.labels = labels
        self.masks  = masks
        print("Image size changed to {}".format(self.images.shape))
        print('Mask not updated')

    def prepare_network(self, run, epoch):
        model = None

        if self.network == 'dir':
            model = directArch(miniXception_loader, self.net_input_shape, objective="malignancy", output_size=self.net_out_size,
                               normalize=self.net_normalize, pooling=self.net_pool)

        elif self.network == 'siam':
            model = siamArch(miniXception_loader, self.net_input_shape, distance='l2', output_size=self.net_out_size, normalize=self.net_normalize,
                             pooling=self.net_pool)

        elif self.network == 'dirR':
            model = directArch(miniXception_loader, self.net_input_shape, objective="rating", output_size=self.net_out_size,
                               normalize=self.net_normalize, pooling=self.net_pool)

        elif self.network == 'siamR':
            model = siamArch(miniXception_loader, self.net_input_shape, distance='l2', output_size=self.net_out_size, normalize=self.net_normalize,
                             pooling=self.net_pool, objective="rating")
        elif self.network == 'trip':
            model = tripArch(miniXception_loader, self.net_input_shape, distance='l2', output_size=self.net_out_size, normalize=self.net_normalize,
                             pooling=self.net_pool, categorize=self.categorize)
        else:
            assert (False)

        w = self.Weights(run=run, epoch=epoch)
        assert (w is not None)

        if self.network == 'dir':
            embed_model = model.extract_core(weights=w, repool=False)
        else:
            embed_model = model.extract_core(weights=w)

        return embed_model

    def run(self, runs, epochs, post, data_subset_id):

        self.prepare_data(data_subset_id)

        for run in runs:
            for epoch in epochs:

                embed_model = self.prepare_network(run=run, epoch=epoch)
                pred = embed_model.predict(self.images)

                out_filename = self.Embed(run, epoch, post)
                pickle.dump((self.images, pred, self.meta, self.labels, self.masks), open(out_filename, 'bw'))
                #pickle.dump(((50*images).astype('int8'), (1000*np.abs(pred)).astype('uint8'), meta, labels, masks.astype('bool')),
                #           open(filename, 'bw'))

                print("Saved: {}".format(out_filename))

                print("Data: images({}), labels({})".format(self.images[0].shape, self.labels.shape))
                print("Image Range = [{:.2f},{:.2f}]".format(np.min(self.images[0]), np.max(self.images[0])))
                print("Embed Range = [{:.2f},{:.2f}]".format(np.min(pred), np.max(pred)))

    def generate_timeline_embedding(self, runs, epochs, post, data_subset_id):

        self.prepare_data(data_subset_id)

        pred = []
        for run in runs:
            for epoch in epochs:
                embed_model = self.prepare_network(run=run, epoch=epoch)
                pred += [np.expand_dims(embed_model.predict(self.images), axis=2)]

        pred = np.concatenate(pred, axis=2)

        return pred