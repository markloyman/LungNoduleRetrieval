from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
try:
    from Network.Common.baseArch import BaseArch
    from Network.dataUtils import get_class_weight
    from Network.Direct.metrics import sensitivity, f1, precision, specificity, root_mean_squared_error, multitask_accuracy
    import Network.FileManager  as File
except:
    from Common.baseArch import BaseArch
    from dataUtils import get_class_weight
    from Direct.metrics import sensitivity, f1, precision, specificity, root_mean_squared_error, multitask_accuracy
    import FileManager as File


class DirectArch(BaseArch):

    def __init__(self, model_loader, input_shape, objective='malignancy',
                 pooling='rmac', output_size=1024, normalize=False, binary=False):

        super().__init__(model_loader, input_shape, objective=objective,
                         pooling=pooling, output_size=output_size, normalize=normalize, binary=binary)

        self.net_type = 'dir' if objective=='malignancy' else 'dirR'
        self.img_input   = Input(shape=input_shape)

        self.base = self.model_loader(input_tensor=self.img_input, input_shape=input_shape,
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
        self.lr         = learning_rate
        self.lr_decay   = decay
        self.model_ready = True

    def set_callbacks(self, label='', gen=False, do_graph=False):
        check = 'accuracy' if self.objective == 'malignancy' else 'loss'
        weight_file_pattern = self.set_weight_file_pattern(check, label)
        callbacks = []
        callbacks += [ModelCheckpoint(weight_file_pattern, monitor='val_loss', save_best_only=False)]
        # on_plateau=ReduceLROnPlateau(monitor='val_loss',factor=0.5,epsilon=0.05,patience=10,min_lr=1e-8,verbose=1)
        #callbacks += [EarlyStopping(monitor='loss', min_delta=0.05, patience=15)]
        if gen:
            callbacks += [TensorBoard(log_dir=self.output_dir + '/logs/' + label, histogram_freq=0, write_graph=do_graph)]
        else:
            callbacks += [TensorBoard(log_dir=self.output_dir + '/logs/' + label, histogram_freq=1, write_graph=do_graph,
                                write_images=False, write_grads=True)]
            # embeddings_freq=5, embeddings_layer_names='n_embedding', embeddings_metadata='meta.tsv')
        return callbacks

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
