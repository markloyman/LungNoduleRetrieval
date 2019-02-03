import pickle
import os
from timeit import default_timer as timer
import numpy as np
from keras.optimizers import Adam
import keras.backend as K
from config import input_dir, output_dir
from Network.Common import losses
import Network.FileManager as File
from Network.dataUtils import crop_center

class BaseArch(object):

    def __init__(self, model_loader, input_shape, objective='malignancy', pooling='rmac', output_size=1024, normalize=False):

        self.model = None
        self.objective = objective
        self.net_type = None
        self.run = None
        self.input_shape = input_shape
        self.output_size = output_size
        self.pooling = pooling
        self.normalize = normalize
        self.model_loader = model_loader
        self.data_ready = False
        self.model_ready = False
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.data_gen = None
        self.last_epoch = None
        self.callbacks = []

    def set_weight_file_pattern(self, check, label):
        if self.data_gen.has_validation():
            weight_file_pattern = '_{{epoch:02d}}-{{{}:.2f}}-{{val_{}:.2f}}'.format(check, check)
        else:
            weight_file_pattern = '_{{epoch:02d}}-{{{}:.2f}}-0'.format(check)
        return self.output_dir + '/Weights/w_' + label + weight_file_pattern + '.h5'

    def load_data(self, images_train, labels_train, images_valid, labels_valid, batch_size=32):
        self.images_train = images_train
        self.labels_train = labels_train
        self.images_valid = images_valid
        self.labels_valid = labels_valid
        self.batch_size = batch_size
        self.data_ready = True

    def load_generator(self, data_gen):
        assert(data_gen.objective == self.objective)
        self.data_gen = data_gen
        if self.regularization_loss:
            self.data_gen.set_regularizer(True)
            #enable_regularization = True

    def compile_model(self, lr, lr_decay, loss, metrics, scheduale=[], temporal_weights=False):
        loss_weights = {}
        do_scheduale = len(scheduale) > 0
        if 'predictions' in loss.keys():
            loss_weights['predictions'] = K.variable(scheduale[0]['weights'][0], name='w_predictions') if do_scheduale else 1.0
        if 'predictions_size' in loss.keys():
            loss_weights['predictions_size'] = K.variable(scheduale[0]['weights'][1], name='w_predictions_size') if do_scheduale else 1.0
        if 'embed_output' in loss.keys():
            loss_weights['embed_output'] = K.variable(scheduale[0]['weights'][1],
                                                          name='w_embed_output') if do_scheduale else 1.0
        if 'distance_matrix' in loss.keys():
            loss_weights['distance_matrix'] = K.variable(scheduale[0]['weights'][1],
                                                          name='w_distance_matrix') if do_scheduale else 1.0

        if self.regularization_loss:
            if 'Dispersion' in self.regularization_loss.keys():
                self.loss['embed_output'] = losses.dispersion_loss
                loss_weights['embed_output'] = K.variable(scheduale[0]['weights'][1], name='w_embed_output') if do_scheduale else self.regularization_loss['Dispersion']
                metrics['embed_output'] = [losses.stdev_loss,
                                           losses.correlation_features_loss_adapter(batch_size=self.embed_size),
                                           losses.correlation_samples_loss_adapter(batch_size=self.embed_size)]
            elif 'Std' in self.regularization_loss.keys():
                loss['embed_output'] = losses.stdev_loss
                loss_weights['embed_output'] = K.variable(scheduale[0]['weights'][0], name='w_embed_output') if do_scheduale else self.regularization_loss['Std']
                metrics['embed_output'] = [losses.dispersion_loss,
                                           losses.correlation_features_loss_adapter(batch_size=self.embed_size),
                                           losses.correlation_samples_loss_adapter(batch_size=self.embed_size)]
            elif 'FeatureCorrelation' in self.regularization_loss.keys():
                loss['embed_output'] = losses.correlation_features_loss_adapter(batch_size=self.embed_size)
                loss_weights['embed_output'] = K.variable(scheduale[0]['weights'][0], name='w_embed_output') if do_scheduale else self.regularization_loss['FeatureCorrelation']
                metrics['embed_output'] = [#losses.dispersion_loss,
                                           #losses.stdev_loss,
                                           losses.correlation_features_loss_adapter(batch_size=self.embed_size),
                                           losses.correlation_samples_loss_adapter(batch_size=self.embed_size)]
            elif 'SampleCorrelation' in self.regularization_loss.keys():
                loss['embed_output'] = losses.correlation_samples_loss_adapter(batch_size=self.embed_size)
                loss_weights['embed_output'] = K.variable(scheduale[0]['weights'][0], name='w_embed_output') if do_scheduale else self.regularization_loss['SampleCorrelation']
                metrics['embed_output'] = [#losses.dispersion_loss,
                                           #losses.stdev_loss,
                                           losses.correlation_samples_loss_adapter(batch_size=self.embed_size),
                                           losses.correlation_features_loss_adapter(batch_size=self.embed_size)]
            else:
                print('[WRN] compile_model: unrecognized regularization loss: {}'.format(self.regularization_loss.keys()))
        else:
            print('No regularization_loss metrics')
            #metrics['embed_output'] = [losses.dispersion_loss,
            #                           losses.stdev_loss,
            #                           losses.correlation_features_loss,
            #                           losses.correlation_samples_loss_adapter(batch_size=self.batch_size)]

        if do_scheduale:
            print("Use loss weight schedualer:\n{}".format(scheduale))
            w_list = [loss_weights[key] for key in ['predictions', 'predictions_size', 'embed_output', 'distance_matrix'] \
                      if key in loss_weights.keys()]
            self.callbacks.append(losses.LossWeightSchedualer(w_list, schedule=scheduale))

        self.model.compile(optimizer=Adam(lr=lr, decay=lr_decay),
                           loss=loss,
                           loss_weights=loss_weights,
                           metrics=metrics,
                           sample_weight_mode='temporal' if temporal_weights else None)
        self.lr = lr
        self.lr_decay = lr_decay
        self.model_ready = True

    def train(self, run='', epoch=0, n_epoch=100, gen=False, do_graph=False):
        self.run = run
        label = self.net_type + run
        callbacks = self.set_callbacks(label=label, gen=gen, do_graph=do_graph)
        start = timer()
        total_time = None
        self.last_epoch = epoch + n_epoch
        if self.lr_decay>0:
            print("LR Decay: {}".format([round(self.lr / (1. + self.lr_decay * n), 5) for n in range(n_epoch)]))
        try:
            if gen:
                self.data_gen.activate_sequences()
                #print("Train Steps: {}, Val Steps: {}".format(self.data_gen.train_N(), self.data_gen.val_N()))
                history = self.model.fit_generator(
                    generator=self.data_gen.training_sequence(),
                    validation_data=self.data_gen.validation_sequence(),
                    initial_epoch=epoch,
                    epochs=self.last_epoch,
                    max_queue_size=9,
                    use_multiprocessing=True if os.name is not 'nt' else False,
                    workers=6 if os.name is not 'nt' else 1,
                    callbacks=callbacks,  # early_stop, on_plateau, early_stop, checkpoint_val, lr_decay, pb
                    verbose=2,
                )
            else:
                history = self.model.fit(
                    x=self.images_train,
                    y=self.labels_train,
                    shuffle=True,
                    validation_data=(self.images_valid, self.labels_valid),
                    batch_size=self.batch_size,
                    initial_epoch=epoch,
                    epochs=epoch+n_epoch,
                    callbacks=callbacks,
                    verbose=2
                )
            total_time = (timer() - start) / 60 / 60
            # history_summarize(history, label)
            pickle.dump(history.history, open(output_dir + '/history/history-{}.p'.format(label), 'bw'))

        finally:
            if total_time is None:
                total_time = (timer() - start) / 60 / 60
            print("Total training time is {:.1f} hours".format(total_time))

    def config_filenames(self, net_type, use_core, keep_spatial_dim=False):
        # init file managers
        Weights = File.Weights(net_type, output_dir=input_dir)
        if use_core:
            if keep_spatial_dim:
                Embed = File.Embed('SP_' + net_type, output_dir=output_dir)
            else:
                Embed = File.Embed(net_type, output_dir=output_dir)
        else:
            if net_type == 'dir':
                Embed = File.Pred(type='malig', pre='dir', output_dir=output_dir)
            elif net_type == 'dirR':
                Embed = File.Pred(type='rating', pre='dirR', output_dir=output_dir)
            elif net_type == 'dirS':
                Embed = File.Pred(type='size', pre='dirS', output_dir=output_dir)
            elif net_type == 'dirRS':
                # assert False # save rating and size in seperate files
                Embed = {}
                Embed['R'] = File.Pred(type='rating', pre='dirRS', output_dir=output_dir)
                Embed['S'] = File.Pred(type='size', pre='dirRS', output_dir=output_dir)
            else:
                print('{} not recognized'.format(net_type))
                assert False

        return Weights, Embed

    def get_data_loader(self, data):
        data_loader = None
        # valid and test got reversed somewhere along the way
        # so i'm forced to keep consistancy
        if data == 'Valid':
            data_loader = self.data_gen.get_flat_valid_data
        elif data == 'Test':
            data_loader = self.data_gen.get_flat_test_data
        elif data == 'Train':
            data_loader = self.data_gen.get_flat_train_data
        else:
            print('{} is not a supported dataset identification'.format(data))

        return data_loader

    def embed(self, epochs, data='Valid', use_core=True, seq_model=False):

        Weights, Embed = self.config_filenames(self.net_type, use_core)

        # get data from generator
        data_loader = self.get_data_loader(data)
        images, labels, classes, masks, meta, conf, size, rating_weights, z = data_loader(resize=False)

        cropped_images = np.array([crop_center(im, msk, size=self.input_shape[0])[0]
                           for im, msk in zip(images, masks)])

        if self.net_type == 'dirS' and not use_core:
            labels = size

        start = timer()
        if self.net_type == 'dirRS' and not use_core:
            embedding = [], []
        else:
            embedding = []
        epochs_done = []

        if use_core:
            embed_model = self.extract_core(repool=False)
        else:
            embed_model = self.model

        for epch in epochs:
            # load weights
            try:
                w = None
                w = Weights(run=self.run, epoch=epch)
                assert(w is not None)
            except:
                print("Skipping. {} not found (w {})".format(epch, w))
                continue

            try:
                if use_core:
                    self.load_core_weights(w)
                else:
                    self.load_weights(w)
                # predict
                if seq_model:
                    pred = np.vstack([embed_model.predict(np.expand_dims(im, axis=0), batch_size=1) for im in cropped_images])
                else:
                    pred = embed_model.predict(cropped_images, batch_size=1)
            except:
                print("Epoch {} failed ({})".format(epch, w))
                continue

            if self.net_type == 'dirRS' and not use_core:
                embedding[0].append(np.expand_dims(pred[0], axis=0))
                embedding[1].append(np.expand_dims(pred[1], axis=0))
            else:
                embedding.append(np.expand_dims(pred, axis=0))
            epochs_done.append(epch)

        if self.net_type == 'dirRS' and not use_core:
            embedding = np.concatenate(embedding[0], axis=0), np.concatenate(embedding[1], axis=0)
        else:
            embedding = np.concatenate(embedding, axis=0)
        total_time = (timer() - start) / 60 / 60
        print("Total training time is {:.1f} hours".format(total_time))

        # dump to Embed file
        if self.net_type == 'dirRS' and not use_core:
            out_filenameR = Embed['R'](self.run, data)
            out_filenameS = Embed['S'](self.run, data)
            pickle.dump((embedding[0], epochs_done, meta, images, classes, labels, masks, conf, rating_weights, z), open(out_filenameR, 'bw'))
            pickle.dump((embedding[1], epochs_done, meta, images, classes, size,   masks, conf, rating_weights, z), open(out_filenameS, 'bw'))
            print("Saved embedding of shape {} to: {}".format(embedding[0].shape, out_filenameR))
            print("Saved embedding of shape {} to: {}".format(embedding[1].shape, out_filenameS))
        else:
            out_filename = Embed(self.run, data)
            pickle.dump((embedding, epochs_done, meta, images, classes, labels, masks, conf, rating_weights, z), open(out_filename, 'bw'))
            print("Saved embedding of shape {} to: {}".format(embedding.shape, out_filename))

    def embed_spatial(self, epochs, data='Valid'):

        Weights, Embed = self.config_filenames(self.net_type, use_core=True, keep_spatial_dim=True)

        # get data from generator
        data_loader = self.get_data_loader(data)
        images, labels, classes, masks, meta, conf, size, rating_weights, z = data_loader()

        start = timer()
        embedding = []
        epochs_done = []
        embed_model = self.extract_spatial_features()

        for epch in epochs:
            # load weights
            try:
                w = None
                w = Weights(run=self.run, epoch=epch)
                assert(w is not None)
            except:
                print("Skipping. {} not found (epoch {})".format(self.run, epch))
                continue

            try:
                self.load_weights(w)
                # predict
                pred = embed_model.predict(images)
            except:
                print("Epoch {} failed ({})".format(epch, w))
                continue

            embedding.append(np.expand_dims(pred, axis=0))
            epochs_done.append(epch)

        embedding = np.concatenate(embedding, axis=0)

        total_time = (timer() - start) / 60 / 60
        print("Total training time is {:.1f} hours".format(total_time))

        # dump to Embed file
        out_filename = Embed(self.run, data)
        pickle.dump((embedding, epochs_done, meta, images, classes, labels, masks, conf, rating_weights, z), open(out_filename, 'bw'))
        print("Saved embedding of shape {} to: {}".format(embedding.shape, out_filename))

    def test(self, images, labels, N=0):
        assert images.shape[0] == labels.shape[0]

        if N == 0:
            N = images.shape[0]

        losses = self.model.evaluate(
                                images[:N],
                                labels[:N],
                                batch_size=32
                            )
        for l,n in zip(losses, self.model.metrics_names):
            print('{}: {}'.format(n,l))

    def predict(self, images, n=0, round=True):
        if n == 0:
            n = images.shape[0]
        predication = \
            self.model.predict(images[:n], batch_size=32)
        if round:
            predication = np.round(predication).astype('uint')

        return predication

    def load_weights(self, w):
        self.model.load_weights(w)
        print("Loaded {}".format(w))

    def load_core_weights(self, w, freeze_n_layers=0):
        import tensorflow as tf
        if type(self.base) == tf.Tensor:
            from keras.models import Model
            model = Model(self.img_input, self.base)
        else:
            model = self.model
        model.load_weights(w, by_name=True)
        if freeze_n_layers > 0:
            for layer in model.layers[:freeze_n_layers]:
                layer.trainable = False
        print("Loaded {}".format(w))

    def compile(self, learning_rate=0.001, decay=0.1, loss='categorical_crossentropy'):
        raise NotImplementedError("compile() is an abstract method")

    def set_callbacks(self, label='', gen=False, do_graph=False):
        raise NotImplementedError("set_callbacks() is an abstract method")

    def extract_core(self, weights=None, repool=False):
        raise NotImplementedError("extract_core() is an abstract method")

    def extract_spatial_features(self, weights=None):
        raise NotImplementedError("extract_spatial_features() is an abstract method")