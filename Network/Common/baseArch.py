import pickle
import os
from timeit import default_timer as timer
import numpy as np

try:
    from Network.dataUtils import get_class_weight
    import Network.FileManager  as File
    output_dir = './output'
    input_dir = './output'
except:
    from dataUtils import get_class_weight
    import FileManager as File
    output_dir = '/output'
    input_dir = '/input'


class BaseArch(object):

    def __init__(self, model_loader, input_shape, objective='malignancy', pooling='rmac', output_size=1024, normalize=False, binary=False):

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

    def embed(self, epochs, data='Valid', use_core=True):
        # init file managers
        Weights = File.Weights(self.net_type, output_dir=input_dir)
        if use_core:
            Embed = File.Embed(self.net_type, output_dir=output_dir)
        else:
            Embed = File.Pred(type='rating', pre='dirR', output_dir=output_dir)

        # get data from generator
        data_loader = None
        # valid and test got reversed somewhere along the way
        # so i'm forced to keep consistancy
        if data == 'Valid':
            data_loader = self.data_gen.get_flat_test_data
        elif data == 'Test':
            data_loader = self.data_gen.get_flat_valid_data
        elif data == 'Train':
            data_loader = self.data_gen.get_flat_train_data
        else:
            print('{} is not a supperted dataset identification'.format(data))
        images, labels, classes, masks, meta, conf = data_loader()

        start = timer()
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
                print("Epoch {} failed ({})".format(epch, w))
                continue

            self.load_weights(w)
            print("Loaded {}".format(w))

            # predict
            pred = embed_model.predict(images)
            embedding.append(np.expand_dims(pred, axis=0))
            epochs_done.append(epch)

        embedding = np.concatenate(embedding, axis=0)
        total_time = (timer() - start) / 60 / 60
        print("Total training time is {:.1f} hours".format(total_time))

        # dump to Embed file
        out_filename = Embed(self.run, data)
        pickle.dump((embedding, epochs_done, meta, images, classes, labels, masks), open(out_filename, 'bw'))
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

    def load_core_weights(self, w):
        self.base.load_weights(w, by_name=True)

    def compile(self, learning_rate=0.001, decay=0.1, loss='categorical_crossentropy'):
        raise NotImplementedError("compile() is an abstract method")

    def set_callbacks(self, label='', gen=False, do_graph=False):
        raise NotImplementedError("set_callbacks() is an abstract method")

    def extract_core(self, weights=None, repool=False):
        raise NotImplementedError("extract_core() is an abstract method")
