import pickle
from timeit import default_timer as timer

import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy

try:
    from Analysis.analysis import history_summarize
    from Network.dataUtils import get_class_weight
    from Network.modelUtils import sensitivity, f1, precision, specificity
    output_dir = './output'
except:
    from dataUtils import get_class_weight
    from modelUtils import sensitivity, f1, precision, specificity
    output_dir = '/output'


class directArch:

    def __init__(self, model_loader, input_shape, classes=2, pooling='rmac', output_size=1024, normalize=False):
    #   input_shape of form: (size, size,1)

        img_input   = Input(shape=input_shape)
        self.base   = model_loader(input_tensor=None, input_shape=input_shape, return_model=True,
                                  pooling=pooling, output_size=output_size, normalize=normalize)
        pred_layer  = Dense(output_size, activation='relu')(self.base(img_input))
        pred_layer  = Dense(classes, activation='softmax', name='predictions')(pred_layer)

        self.model = Model(img_input, pred_layer, name='directArch')
        self.data_ready  = False
        self.model_ready = False

    def compile(self, learning_rate = 0.001, decay=0.1):
        categorical_accuracy.__name__ = 'accuracy'
        sensitivity.__name__ = 'recall'
        self.model.compile( optimizer   = Adam(lr=learning_rate, decay=decay),
                            loss        = 'categorical_crossentropy', #'categorical_crossentropy', mean_squared_error
                            metrics     = [categorical_accuracy, f1, sensitivity, precision, specificity] )
        self.model_ready = True

    def load_data(self, images_train, labels_train, images_valid, labels_valid, batch_sz=32):
        self.images_train = images_train
        self.labels_train = labels_train
        self.images_valid = images_valid
        self.labels_valid = labels_valid
        self.batch_sz = batch_sz
        self.data_ready = True

    def load_generator(self, data_gen):
        self.data_gen = data_gen

    def train(self, label='', epoch=0, n_epoch=100, gen = False):

        checkpoint = ModelCheckpoint(output_dir+'/Weights/w_' + label + '_{epoch:02d}-{accuracy:.2f}-{val_accuracy:.2f}.h5',
                                           monitor='val_loss', save_best_only=False)
        #on_plateau       = ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=0.05, patience=10, min_lr=1e-8, verbose=1)
        early_stop       = EarlyStopping(monitor='loss', min_delta=0.05, patience=15)
        if gen:
            board            = TensorBoard(log_dir=output_dir+'/logs', histogram_freq=0, write_graph=False)
        else:
            board = TensorBoard(log_dir=output_dir + '/logs', histogram_freq=1, write_graph=True,
                                write_images=True, batch_size=64, write_grads=True)
                                #embeddings_freq=5, embeddings_layer_names='n_embedding', embeddings_metadata='meta.tsv')
        start = timer()
        total_time = None
        try:
            if gen:
                print("Train Steps: {}, Val Steps: {}".format(self.data_gen.train_N(), self.data_gen.val_N()))
                history = self.model.fit_generator(
                    generator =  self.data_gen.next_train(),
                    steps_per_epoch= self.data_gen.train_N(),
                    #class_weight = class_weight,
                    validation_data  = self.data_gen.next_val(),
                    validation_steps = self.data_gen.val_N(),
                    initial_epoch = epoch,
                    epochs = epoch+n_epoch,
                    max_queue_size=3,
                    callbacks = [checkpoint, board], # early_stop, on_plateau, early_stop, checkpoint_val, lr_decay, pb
                    verbose = 2
                )
            else:
                class_weight = get_class_weight(self.labels_train)
                history = self.model.fit(
                    x = self.images_train,
                    y = self.labels_train,
                    shuffle = True,
                    #class_weight = class_weight,
                    validation_data = (self.images_valid, self.labels_valid),
                    batch_size = self.batch_sz,
                    initial_epoch=epoch,
                    epochs = epoch+n_epoch,
                    callbacks = [checkpoint, early_stop, board],
                    verbose = 2
                )
            total_time = (timer() - start) / 60 / 60
            # history_summarize(history, label)
            pickle.dump(history.history, open(output_dir+ '/history/history-{}.p'.format(label), 'bw'))

        finally:
            if total_time is None:
                total_time = (timer() - start) / 60 / 60
            print("Total training time is {:.1f} hours".format(total_time))

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
        self.model.load_weights('Weights/'+w)
        return None

    def extract_core(self):
        return self.base