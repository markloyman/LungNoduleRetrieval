import pickle
from timeit import default_timer as timer
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
try:
    from Network.dataUtils import get_class_weight
    from Network.Direct.metrics import sensitivity, f1, precision, specificity, root_mean_squared_error, multitask_accuracy
    import Network.FileManager  as File
    output_dir = './output'
except:
    from dataUtils import get_class_weight
    from Direct.metrics import sensitivity, f1, precision, specificity, root_mean_squared_error, multitask_accuracy
    import FileManager as File
    output_dir = '/output'


class directArch:

    def __init__(self, model_loader, input_shape, objective='malignancy', pooling='rmac', output_size=1024, normalize=False, binary=False):
    #   input_shape of form: (size, size,1)

        self.objective = objective
        self.img_input   = Input(shape=input_shape)
        self.input_shape = input_shape
        self.output_size = output_size
        self.pooling = pooling
        self.normalize = normalize
        self.model_loader = model_loader
        self.base = model_loader(input_tensor=self.img_input, input_shape=input_shape,
                                 pooling=pooling, output_size=output_size, normalize=normalize, binary=binary)
        if objective=='malignancy':
            pred_layer = Dense(2, activation='softmax', name='predictions')(self.base)
        elif objective=='rating':
            pred_layer = Dense(9, activation='linear', name='predictions')(self.base)
        else:
            print("ERR: Illegual objective given ({})".format(objective))
            assert(False)

        self.model = Model(self.img_input, pred_layer, name='directArch')

        self.input_shape = input_shape
        self.data_ready  = False
        self.model_ready = False

    def compile(self, learning_rate = 0.001, decay=0.1, loss = 'categorical_crossentropy'):
        categorical_accuracy.__name__ = 'accuracy'
        sensitivity.__name__ = 'recall'
        root_mean_squared_error.__name__ = 'rmse'
        multitask_accuracy.__name__ = 'accuracy'
        metrics = []
        if self.objective == 'malignancy':
            metrics = [categorical_accuracy, f1, sensitivity, precision, specificity]
        elif self.objective == 'rating':
            metrics = [root_mean_squared_error, multitask_accuracy]
        self.model.compile( optimizer   = Adam(lr=learning_rate, decay=decay),
                            loss        = loss, #'categorical_crossentropy', mean_squared_error
                            metrics     = metrics )
        self.model_ready = True

    def load_data(self, images_train, labels_train, images_valid, labels_valid, batch_sz=32):
        self.images_train = images_train
        self.labels_train = labels_train
        self.images_valid = images_valid
        self.labels_valid = labels_valid
        self.batch_sz = batch_sz
        self.data_ready = True

    def load_generator(self, data_gen):
        assert(data_gen.objective == self.objective)
        self.data_gen = data_gen

    def set_callbacks(self, label='', gen=False, do_graph=False):
        callbacks = []
        check = 'accuracy' if self.objective == 'malignancy' else 'loss'
        if self.data_gen.val_factor > 0:
            weight_file_pattern = '_{{epoch:02d}}-{{{}:.2f}}-{{val_{}:.2f}}'.format(check, check)
        else:
            weight_file_pattern = '_{{epoch:02d}}-{{{}:.2f}}-0'.format(check)
        callbacks += [ModelCheckpoint(
            output_dir + '/Weights/w_' + label + weight_file_pattern + '.h5',
            monitor='val_loss', save_best_only=False)]
        # on_plateau=ReduceLROnPlateau(monitor='val_loss',factor=0.5,epsilon=0.05,patience=10,min_lr=1e-8,verbose=1)
        callbacks += [EarlyStopping(monitor='loss', min_delta=0.05, patience=15)]
        if gen:
            callbacks += [TensorBoard(log_dir=output_dir + '/logs/' + label, histogram_freq=0, write_graph=do_graph)]
        else:
            callbacks += [TensorBoard(log_dir=output_dir + '/logs/' + label, histogram_freq=1, write_graph=do_graph,
                                write_images=False, write_grads=True)]
            # embeddings_freq=5, embeddings_layer_names='n_embedding', embeddings_metadata='meta.tsv')
        return callbacks

    def train(self, label='', epoch=0, n_epoch=100, gen=False, do_graph=False):
        callbacks = self.set_callbacks(label=label, gen=gen, do_graph=do_graph)
        start = timer()
        total_time = None
        self.last_epoch = epoch + n_epoch
        try:
            if gen:
                print("Train Steps: {}, Val Steps: {}".format(self.data_gen.train_N(), self.data_gen.val_N()))
                history = self.model.fit_generator(
                    generator=self.data_gen.next_train(),
                    steps_per_epoch=self.data_gen.train_N(),
                    validation_data=self.data_gen.next_val(),
                    validation_steps=self.data_gen.val_N(),
                    initial_epoch=epoch,
                    epochs=self.last_epoch,
                    max_queue_size=3,
                    callbacks=callbacks,  # early_stop, on_plateau, early_stop, checkpoint_val, lr_decay, pb
                    verbose=2
                )
            else:
                history = self.model.fit(
                    x=self.images_train,
                    y=self.labels_train,
                    shuffle=True,
                    validation_data=(self.images_valid, self.labels_valid),
                    batch_size=self.batch_sz,
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

    def embed(self, epoch0=0, delta_epoch=5):
        # init file managers
        Weights = File.Weights(self.net_type)
        Embed = File.Embed(self.net_type)

        # get data from generator
        images, labels, classes, masks, meta = self.data_gen.get_valid_data()

        epochs = list(range(epoch0, self.last_epoch, delta_epoch))
        embedding = []
        for epch in epochs:
            # load weights
            w = Weights(run=self.run, epoch=epch)
            assert(w is not None)
            embed_model = self.extract_core(weights=w, repool=False)
            print("Loaded {}".format(w))

            # predict
            pred = embed_model.predict(images)
            embedding.append(np.expand_dims(pred, axis=0))

        embedding = np.concatenate(embedding, axis=0)

        # dump to Embed file
        out_filename = Embed(self.run, 'Valid')
        pickle.dump((embedding, epochs, meta, images, classes, labels, masks), open(out_filename, 'bw'))
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

    def predict(self, images, n=0, round = True):
        if n == 0:
            n = images.shape[0]
        predication = \
            self.model.predict(images[:n], batch_size=32)
        if round:
            predication = np.round(predication).astype('uint')

        return predication


    def load_weights(self, w):
        self.model.load_weights(w)
        return None

    def extract_core(self, weights=None, repool=False):
        if repool:
            from keras.layers import GlobalAveragePooling2D
            from keras.layers import MaxPooling2D
            from keras.layers import Lambda
            import keras.backend as K
            no_pool_model = Model(inputs=self.img_input,
                                  outputs=Dense(2, activation='softmax', name='predictions')(self.model_loader(input_tensor=self.img_input, input_shape=self.input_shape,
                                  pooling='none', output_size=self.output_size, normalize=self.normalize)),
                                  name='no-pool')
            no_pool_model.load_weights(weights)
            no_pool_model.layers.pop() # remove dense
            no_pool_model.layers.pop() # remove normal
            no_pool_model.layers.pop()  # remove flatten
            x = no_pool_model.layers[-1].output
            if self.pooling == 'rmac':
                x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='embed_pool')(x)
                x = GlobalAveragePooling2D(name='embeding')(x)
            else:
                assert(False)
            if self.normalize:
                x = Lambda(lambda q: K.l2_normalize(q, axis=-1), name='n_embedding')(x)
            model = Model(no_pool_model.input, x)

        else:
            if weights is not None:
                self.model.load_weights(weights)
            model = Model(  inputs  = self.img_input,
                        outputs = self.base,
                        name    ='directEmbed')
        return model
