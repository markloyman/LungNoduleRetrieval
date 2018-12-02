from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.layers import Input
from keras.layers import Lambda, Activation
from keras.models import Model

try:
    from Network.Common.baseArch import BaseArch
    from Network.Common.distances import euclidean_distance, l1_distance, cosine_distance, distance_output_shape
    from Network.Triplet.metrics import triplet_margin, rank_accuracy, kendall_correlation
    from Network import FileManager as File
except:
    from Common.baseArch import BaseArch
    from Common.distances import euclidean_distance, l1_distance, cosine_distance, distance_output_shape
    from Triplet.metrics import triplet_margin, rank_accuracy, kendall_correlation
    import FileManager as File


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


class TripArch(BaseArch):

    def __init__(self, model_loader, input_shape, objective='malignancy',
                 pooling='rmac', output_size=1024, distance='l2', normalize=False, categorize=False, regularization_loss={}):

        super().__init__(model_loader, input_shape, objective=objective,
                         pooling=pooling, output_size=output_size, normalize=normalize)

        self.net_type = 'tripR' if (objective == 'rating') else 'trip'

        img_input_ref  = Input(shape=input_shape)
        img_input_pos  = Input(shape=input_shape)
        img_input_neg  = Input(shape=input_shape)

        self.base =  self.model_loader(input_tensor=None, input_shape=input_shape, return_model=True,
                                  pooling=pooling, output_size=output_size, normalize=normalize)
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
            assert False

        distance_layer_pos = Lambda(distance_layer,
                                    output_shape=distance_output_shape, name='pos_dist')([base_ref, base_pos])
        distance_layer_neg = Lambda(distance_layer,
                                    output_shape=distance_output_shape, name='neg_dist')([base_ref, base_neg])

        trip_layer = Lambda(lambda vects: K.concatenate(vects, axis=1), name='output')([distance_layer_pos, distance_layer_neg])

        embed = Lambda(lambda x: K.concatenate([x[0], x[1], x[2]], axis=0), name='embed_output')([base_ref, base_pos, base_neg])

        self.categorize = categorize
        if categorize:
            trip_layer = Activation('softmax')(trip_layer)

        self.regularization_loss = regularization_loss

        outputs = []
        outputs.append(trip_layer)
        if regularization_loss:
            outputs.append(embed)

        self.model =    Model(  inputs=[img_input_ref, img_input_pos, img_input_neg],
                                outputs = outputs,
                                name='tripletArch')

    def compile(self, learning_rate = 0.001, decay=0.0, loss = 'mean_squared_error', scheduale=[]):
        rank_accuracy.__name__ = 'accuracy'
        kendall_correlation.__name__  = 'corr'

        if self.categorize:
            loss = {'output': 'categorical_crossentropy'}
        else:
            loss =  {'output': triplet_loss} #[contrastive_loss_p, contrastive_loss_n] #'mean_squared_error' #triplet_loss
        metrics = {'output': [rank_accuracy, kendall_correlation]}

        self.compile_model(lr=learning_rate, lr_decay=decay,
                           loss=loss,
                           metrics=metrics,
                           scheduale=scheduale)
        # lr = self.lr * (1. / (1. + self.decay * self.iterations))
        self.lr         = learning_rate
        self.lr_decay   = decay
        self.model_ready = True

    def set_callbacks(self, label='', gen=False, do_graph=False):
        assert do_graph is False
        check = 'loss'
        weight_file_pattern = self.set_weight_file_pattern(check, label)
        callbacks = self.callbacks
        callbacks += [ModelCheckpoint(weight_file_pattern, monitor='val_loss', save_best_only=False)]
        #on_plateau      = ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=1e-2, patience=5, min_lr=1e-6, verbose=1)
        #early_stop      = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5)
        #lr_decay        = LearningRateScheduler(self.scheduler)
        if gen:
            callbacks += [TensorBoard(log_dir=self.output_dir+'/logs/'+label, histogram_freq=0, write_graph=False)]
        else:
            callbacks += [TensorBoard(log_dir=self.output_dir+'/logs'+label, histogram_freq=1, write_graph=True,
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
