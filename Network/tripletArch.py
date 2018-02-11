import pickle
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input
from keras.layers import Lambda, Activation
from keras.models import Model
from keras.optimizers import Adam

try:
    from Network.metrics import triplet_margin, rank_accuracy, kendall_correlation
    output_dir = './output'
except:
    from metrics import triplet_margin, rank_accuracy, kendall_correlation
    output_dir = '/output'


def huber(a, d):
    return K.square(d)*(K.sqrt(1+K.square(a/d)) - 1.0)


def huber_inv(a, d=1.0):
    return K.sqrt(d)*(K.square(1+K.sqrt(a/d)) - 1.0)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def l1_distance(vects):
    x, y = vects
    return K.sum(K.abs(x - y), axis=1, keepdims=True)


def cosine_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return 1.0 - K.batch_dot(x, y, axes=-1)


def trip_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss_p(y_true, y_pred):
    return K.square(y_pred)


def contrastive_loss_n(y_true, y_pred):
    return K.square(K.maximum(1 - y_pred, 0))


def triplet_loss(_, y_pred):
    '''
        Assume: y_pred shape is (batch_size, 2)
    '''
    margin = K.constant(triplet_margin)

    subtraction = K.constant([1, -1], shape=(2, 1))
    diff =  K.dot(K.square(y_pred), subtraction)

    #loss = K.maximum(K.constant(0), margin + diff)
    loss = K.softplus(diff)

    return loss


class printbatch(Callback):
    def on_batch_end(self, epoch, logs={}):
        print('E#{}: {}'.format(epoch, logs))

class tripArch:

    def __init__(self, model_loader, input_shape, pooling='rmac', output_size=1024, distance='l2', normalize=False, binary=False, categorize=False):
    #   input_shape of form: (size, size,1)

        self.categorize = categorize

        img_input_ref  = Input(shape=input_shape)
        img_input_pos  = Input(shape=input_shape)
        img_input_neg  = Input(shape=input_shape)
        self.input_shape = input_shape

        self.base =  model_loader(input_tensor=None, input_shape=input_shape, return_model=True,
                                  pooling=pooling, output_size=output_size, normalize=normalize, binary=binary)
        #self.base.summary()
        base_ref = self.base(img_input_ref)
        base_pos = self.base(img_input_pos)
        base_neg = self.base(img_input_neg)

        if distance == 'l2':
            distance_layer = euclidean_distance
        elif distance == 'l1':
            distance_layer = l1_distance
        elif distance == 'cosine':
            distance_layer = cosine_distance
        else:
            assert(False)

        distance_layer_pos = Lambda(distance_layer,
                                    output_shape=trip_dist_output_shape, name='pos_dist')([base_ref, base_pos])
        distance_layer_neg = Lambda(distance_layer,
                                    output_shape=trip_dist_output_shape, name='neg_dist')([base_ref, base_neg])

        output_layer = Lambda(lambda vects: K.concatenate(vects, axis=1), name='output')([distance_layer_pos, distance_layer_neg])
        if categorize:
            output_layer = Activation('softmax')(output_layer)

        self.model =    Model(  inputs=[img_input_ref, img_input_pos, img_input_neg],
                                outputs = output_layer,
                                name='tripletArch')

        self.data_ready  = False
        self.model_ready = False

    def compile(self, learning_rate = 0.001, decay=0.1, loss = 'mean_squared_error'):
        rank_accuracy.__name__ = 'accuracy'
        kendall_correlation.__name__  = 'corr'

        if self.categorize:
            loss = 'categorical_crossentropy'
        else:
            loss =  triplet_loss #[contrastive_loss_p, contrastive_loss_n] #'mean_squared_error' #triplet_loss
        metrics = [rank_accuracy, kendall_correlation]

        self.model.compile( optimizer   = Adam(lr=learning_rate, decay=decay), #, decay=0.01*learning_rate),
                            loss        = loss,
                            metrics     = metrics)
        # lr = self.lr * (1. / (1. + self.decay * self.iterations))
        self.lr         = learning_rate
        self.lr_decay   = decay
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

    def train(self, label='', epoch=0, n_epoch=100, gen = False):
        if self.lr_decay>0:
            print("LR Decay: {}".format([round(self.lr / (1. + self.lr_decay * n), 5) for n in range(n_epoch)]))

        checkpoint      = ModelCheckpoint(output_dir+'/Weights/w_' + label + '_{epoch:02d}-{loss:.2f}-{val_loss:.2f}.h5',
                                         monitor='loss', save_best_only=False)
        #checkpoint_val  = ModelCheckpoint('./Weights/w_'+label+'_{epoch:02d}-{loss:.2f}-{val_loss:.2f}.h5',
        #                                   monitor='val_loss', save_best_only=True)
        #on_plateau      = ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=1e-2, patience=5, min_lr=1e-6, verbose=1)
        #early_stop      = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5)
        #lr_decay        = LearningRateScheduler(self.scheduler)
        if gen:
            print("DataGen, so lightweight TensorBoard")
            board = TensorBoard(log_dir=output_dir+'/logs', histogram_freq=0, write_graph=False)
        else:
            print("Extended TensorBoard")
            #board = TensorBoard(log_dir=output_dir+'/logs', histogram_freq=1, write_graph=True,
            #                    write_images=True, batch_size=32, write_grads=True)
            board = TensorBoard(log_dir=output_dir + '/logs', histogram_freq=1, write_graph=True, write_grads=True,
                                                    write_images=True)

        pb = printbatch()

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
                    callbacks = [checkpoint, board ], # early_stop, on_plateau, early_stop, checkpoint_val, lr_decay, pb
                    verbose = 2
                )

                #next_batch = self.data_gen.next_val().__next__()
                #pred = self.model.evaluate(next_batch)
                #pickle.dump( (next_batch,pred), open('tmp.p', 'bw') )

            else:
                history = self.model.fit(
                    x = [self.images_train[0], self.images_train[1], self.images_train[2]],
                    y =  self.labels_train,
                    shuffle = True,
                    #class_weight = class_weight,
                    validation_data = ([self.images_valid[0], self.images_valid[1], self.images_valid[2]], self.labels_valid),
                    batch_size = 32,
                    initial_epoch = epoch,
                    epochs = epoch+n_epoch,
                    callbacks = [checkpoint, board], # on_plateau, early_stop,checkpoint_val
                    verbose = 2
                )

            total_time = (timer() - start) / 60 / 60

            #history_summarize(history, label)
            pickle.dump(history.history, open(output_dir+'/history/history-{}.p'.format(label), 'bw'))

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
