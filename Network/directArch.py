import pickle
from timeit import default_timer as timer

import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from Analysis.analysis import history_summarize
from Network.dataUtils import get_class_weight
from Network.modelUtils import sensitivity, specificity, precision


class directArch:

    def __init__(self, model_loader, input_shape, classes=2):
    #   input_shape of form: (size, size,1)

        img_input   = Input(shape=input_shape)
        self.base   = model_loader(input_tensor=img_input, input_shape=input_shape, pooling='rmac')
        pred_layer  = Dense(classes, activation='softmax', name='predictions')(self.base)

        self.model = Model(img_input, pred_layer, name='directArch')
        self.data_ready  = False
        self.model_ready = False

    def compile(self, learning_rate):
        self.model.compile( optimizer   = Adam(lr=learning_rate, decay=0.01*learning_rate),
                            loss        = 'categorical_crossentropy',
                            metrics     = ['categorical_accuracy', sensitivity, specificity, precision] )
        self.model_ready = True

    def load_data(self, images_train, labels_train, images_valid, labels_valid):
        self.images_train = images_train
        self.labels_train = labels_train
        self.images_valid = images_valid
        self.labels_valid = labels_valid

        self.data_ready = True

    def train(self, label=''):

        model_checkpoint = ModelCheckpoint('w_'+label+'_{epoch:02d}-{categorical_accuracy:.2f}-{val_categorical_accuracy:.2f}.h5',
                                           monitor='val_loss', save_best_only=True)

        on_plateau       = ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=0.05, patience=10, min_lr=1e-8, verbose=1)
        early_stop       = EarlyStopping(monitor='loss', min_delta=0.05, patience=15)
        board            = TensorBoard(log_dir='./logs', histogram_freq=1)

        class_weight     = get_class_weight(self.labels_train)

        start = timer()
        total_time = None
        try:
            history = self.model.fit(
                x = self.images_train,
                y = self.labels_train,
                shuffle = True,
                class_weight = class_weight,
                validation_data = (self.images_valid, self.labels_valid),
                batch_size = 32,
                epochs = 100,
                callbacks = [model_checkpoint, on_plateau, early_stop, board],
                verbose = 2
            )

            total_time = (timer() - start) / 60 / 60

            history_summarize(history, label)
            pickle.dump(history.history, open('history-{}.p'.format(label), 'bw'))

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