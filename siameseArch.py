import numpy as np
import pickle
from timeit import default_timer as timer
import random

from keras.models import Model
from keras import layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Lambda
from keras.optimizers import Adam, Nadam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, LearningRateScheduler
from keras import backend as K

from analysis import history_summarize
from modelUtils import sensitivity, specificity, precision, binary_accuracy
from dataUtils import get_class_weight

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
    #return (1/32)*K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 5
    return K.mean( (1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))
    #return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

class siamArch:

    def __init__(self, model_loader, input_shape, classes=2):
    #   input_shape of form: (size, size,1)

        img_input1  = Input(shape=input_shape)
        img_input2  = Input(shape=input_shape)
        self.input_shape = input_shape

        self.base =  model_loader(input_tensor=None, input_shape=input_shape, pooling='rmac', return_model=True)
        base1 = self.base(img_input1)
        base2 = self.base(img_input2)

        distance_layer = Lambda(euclidean_distance,
                                output_shape=eucl_dist_output_shape)([base1, base2])

        self.model =    Model(  inputs=[img_input1,img_input2],
                                outputs = distance_layer,
                                name='siameseArch')

        self.data_ready  = False
        self.model_ready = False

    def compile(self, learning_rate = 0.001):
        self.model.compile( optimizer   = Adam(lr=learning_rate), #, decay=0.01*learning_rate),
                            loss        = contrastive_loss,
                            metrics     = [binary_accuracy]) #['binary_accuracy', 'categorical_accuracy', sensitivity, specificity, precision] )
        # lr = self.lr * (1. / (1. + self.decay * self.iterations))
        self.model_ready = True

    def load_data(self, images_train, labels_train, images_valid, labels_valid):
        self.images_train = images_train
        self.labels_train = labels_train
        self.images_valid = images_valid
        self.labels_valid = labels_valid

        self.data_ready = True

    def load_generator(self, data_gen):
        self.data_gen = data_gen

    def scheduler(self, epoch):
        if epoch % 2 == 0 and epoch != 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * .9)
            print("lr changed to {}".format(lr * .9))
        return K.get_value(self.model.optimizer.lr)

    def train(self, label='', epoch=0, n_epoch=100, gen = False, new_lr = None):
        checkpoint      = ModelCheckpoint('./Weights/w_' + label + '_{epoch:02d}-{loss:.2f}-{val_loss:.2f}.h5',
                                         monitor='loss', save_best_only=False)
        #checkpoint_val  = ModelCheckpoint('./Weights/w_'+label+'_{epoch:02d}-{loss:.2f}-{val_loss:.2f}.h5',
        #                                   monitor='val_loss', save_best_only=True)
        #on_plateau      = ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=0.02, patience=20, min_lr=1e-8, verbose=1)
        #early_stop      = EarlyStopping(monitor='loss', min_delta=0.01, patience=30)
        lr_decay        = LearningRateScheduler(self.scheduler)
        board           = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)


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
                    callbacks = [checkpoint, lr_decay, board], # on_plateau, early_stop, checkpoint_val
                    verbose = 2
                )
            else:
                history = self.model.fit(
                    x = [self.images_train[0],self.images_train[1]],
                    y =  self.labels_train,
                    shuffle = True,
                    #class_weight = class_weight,
                    validation_data = ([self.images_valid[0],self.images_valid[1]], self.labels_valid),
                    batch_size = 32,
                    initial_epoch = epoch,
                    epochs = epoch+n_epoch,
                    callbacks = [checkpoint, board], # on_plateau, early_stop,checkpoint_val
                    verbose = 2
                )

            total_time = (timer() - start) / 60 / 60

            #history_summarize(history, label)
            pickle.dump(history.history, open('./history/history-{}.p'.format(label), 'bw'))

        finally:
            if total_time is None:
                total_time = (timer() - start) / 60 / 60
            print("Total training time is {:.1f} hours".format(total_time))

    def test(self, images, labels, N=0):
        assert images.shape[0] == labels.shape[0]

        if N == 0:
            N = images.shape[0]

        losses = self.model.evaluate(
                                [images[0][:N],images[1][:N]],
                                labels[:N],
                                batch_size=32
                            )
        for l,n in zip(losses, self.model.metrics_names):
            print('{}: {}'.format(n,l))

    def predict(self, images, n=0, round = True):
        if n == 0:
            n = images[0].shape[0]
        predication = \
            self.model.predict([images[0][:n],images[1][:n]], batch_size=32)
        if round:
            predication = np.round(predication).astype('uint')

        return predication


    def load_weights(self, w):
        self.model.load_weights(w)
        return None

    def extract_core(self, weights=None):
        if weights is not None:
            self.model.load_weights(weights)

        img_input = Input(shape=self.input_shape)
        model = Model(  inputs  = img_input,
                        outputs = self.base(img_input),
                        name    ='siameseEmbed')

        return model