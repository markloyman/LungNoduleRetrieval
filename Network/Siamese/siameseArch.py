from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.layers import Input
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import Adam

try:
    from Network.Common.baseArch import BaseArch
    from Network.Common.distances import euclidean_distance, l1_distance, cosine_distance, distance_output_shape
    from Network.Siamese.metrics import siamese_margin, binary_accuracy, binary_precision_inv, binary_recall_inv, binary_f1_inv, binary_assert, pearson_correlation
    from Network import FileManager as File
except:
    from Common.baseArch import BaseArch
    from Common.distances import euclidean_distance, l1_distance, cosine_distance, distance_output_shape
    from Siamese.metrics import siamese_margin, binary_accuracy, binary_precision_inv, binary_recall_inv, binary_f1_inv, binary_assert, pearson_correlation
    import FileManager as File


def contrastive_loss(y_true, y_pred, marginal=False):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = siamese_margin
    if marginal:
        return (1 - y_true) * K.square(K.maximum(y_pred - 0.25 * margin, 0)) \
               + y_true * K.square(K.maximum(margin - y_pred, 0))
    else:
        return (1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0))



class SiamArch(BaseArch):

    def __init__(self, model_loader, input_shape, objective="malignancy",
                 pooling='rmac', output_size=1024, distance='l2', normalize=False):

        super().__init__(model_loader, input_shape, objective=objective,
                         pooling=pooling, output_size=output_size, normalize=normalize, binary=False)

        self.net_type = 'siam' if objective == 'malignancy' else 'siamR'

        img_input1 = Input(shape=input_shape)
        img_input2 = Input(shape=input_shape)

        self.base = self.model_loader(input_tensor=None, input_shape=input_shape, return_model=True,
                                  pooling=pooling, output_size=output_size, normalize=normalize)
        #self.base.summary()
        base1 = self.base(img_input1)
        base2 = self.base(img_input2)

        if distance == 'l2':
            distance_layer = Lambda(euclidean_distance,
                                    output_shape=distance_output_shape)([base1, base2])
        elif distance == 'l1':
            distance_layer = Lambda(l1_distance,
                                    output_shape=distance_output_shape)([base1, base2])
        elif distance == 'cosine':
            distance_layer = Lambda(cosine_distance,
                                    output_shape=distance_output_shape)([base1, base2])
        else:
            assert(False)

        self.model =    Model(  inputs=[img_input1,img_input2],
                                outputs = distance_layer,
                                name='siameseArch')

    def compile(self, learning_rate = 0.001, decay=0.1, loss = 'mean_squared_error'):
        binary_accuracy.__name__      = 'accuracy'
        binary_precision_inv.__name__ = 'precision'
        binary_recall_inv.__name__    = 'recall'
        binary_f1_inv.__name__        = 'f1'
        pearson_correlation.__name__  = 'corr'

        if self.objective == "malignancy":
            loss = contrastive_loss
            metrics = [binary_accuracy, binary_f1_inv, binary_precision_inv, binary_recall_inv]
            #['binary_accuracy', 'categorical_accuracy', sensitivity, specificity, precision] )
        elif self.objective == "rating":
            metrics = [pearson_correlation]
        else:
            print("ERR: {} is not a valid objective".format(self.objective))
            assert (False)

        self.model.compile( optimizer   = Adam(lr=learning_rate, decay=decay), #, decay=0.01*learning_rate),
                            loss        = loss,
                            metrics     = metrics)
        # lr = self.lr * (1. / (1. + self.decay * self.iterations))
        self.lr         = learning_rate
        self.lr_decay   = decay
        self.model_ready = True

    def scheduler(self, epoch):
        if epoch % 2 == 0 and epoch != 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * .9)
            print("lr changed to {}".format(lr * .9))
        return K.get_value(self.model.optimizer.lr)

    def set_callbacks(self, label='', gen=False, do_graph=False):
        assert do_graph is False
        check = 'loss'
        weight_file_pattern = self.set_weight_file_pattern(check, label)
        callbacks = []
        callbacks += [ModelCheckpoint(weight_file_pattern, monitor='val_loss', save_best_only=False)]
        #on_plateau      = ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=1e-2, patience=5, min_lr=1e-6, verbose=1)
        #early_stop      = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5)
        #lr_decay        = LearningRateScheduler(self.scheduler)
        if gen:
            callbacks += [TensorBoard(log_dir=self.output_dir+'/logs/' + label, histogram_freq=0, write_graph=False)]
        else:
            callbacks += [TensorBoard(log_dir=self.output_dir+'/logs/' + label, histogram_freq=1, write_graph=True,
                                write_images=False, write_grads=True)]
        return callbacks

    def extract_core(self, weights=None, repool=False):
        assert repool is False
        if weights is not None:
            self.model.load_weights(weights)

        img_input = Input(shape=self.input_shape)
        model = Model(  inputs  = img_input,
                        outputs = self.base(img_input),
                        name    ='siameseEmbed')
        return model
