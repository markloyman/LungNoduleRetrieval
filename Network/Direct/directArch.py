from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Concatenate, BatchNormalization, Activation
try:
    from Network.Common.baseArch import BaseArch
    #from Network.Common import losses
    from Network.dataUtils import get_class_weight
    from Network.Direct.metrics import sensitivity, f1, precision, specificity, root_mean_squared_error, multitask_accuracy
    import Network.FileManager  as File
except:
    from Common.baseArch import BaseArch
    #from Common import losses
    from dataUtils import get_class_weight
    from Direct.metrics import sensitivity, f1, precision, specificity, root_mean_squared_error, multitask_accuracy
    import FileManager as File


class DirectArch(BaseArch):

    def __init__(self, model_loader, input_shape, objective='malignancy',
                 pooling='rmac', output_size=1024, normalize=False, binary=False, l1_regularization=None,
                 batch_size=32, regularization_loss={}, separated_prediction=False):

        super().__init__(model_loader, input_shape, objective=objective,
                         pooling=pooling, output_size=output_size, normalize=normalize)

        if objective == 'malignancy':
            self.net_type = 'dir'
        elif objective in 'rating':
            self.net_type = 'dirR'
        elif objective == 'size':
            self.net_type = 'dirS'
        elif objective == 'rating_size':
            self.net_type = 'dirRS'
        elif objective == 'distance-matrix':
            self.net_type = 'dirD'
        else:
            print("{} objective not recognized".format(objective))
            assert False

        self.img_input   = Input(shape=input_shape)

        self.batch_size = batch_size
        self.embed_size = batch_size

        self.base = self.model_loader(input_tensor=self.img_input, input_shape=input_shape,
                                      pooling=pooling, output_size=output_size,
                                      normalize=normalize, binary=binary, regularize=l1_regularization)

        self.regularization_loss = regularization_loss

        outputs = []
        if objective=='malignancy':
            pred_layer = Dense(2, activation='softmax', name='predictions')(self.base)
            outputs.append(pred_layer)
        elif objective=='rating':
            if separated_prediction:
                p = [[] for _ in range(9)]
                for i in range(9):
                    p[i] = Dense(output_size // 2, activation='relu', name='dense0_{}'.format(i))(self.base)
                    p[i] = Dense(output_size // 4, activation='relu', name='dense1_{}'.format(i))(p[i])
                    p[i] = Dense(1, activation='linear', name='pred_{}'.format(i))(p[i])
                pred_layer = Concatenate(axis=1, name='predictions')(p)
            else:
                pred_layer = Dense(9, activation='linear', name='predictions')(self.base)
            outputs.append(pred_layer)
        elif objective == 'size':
            #x = Dense(output_size // 2, name='dense_size')(self.base)
            #x = BatchNormalization(name='dense_size_bn')(x)
            #x = Activation('relu', name='dense_size_act')(x)
            size_layer = Dense(1, activation='linear', name='predictions_size')(self.base)
            outputs.append(size_layer)
        elif objective == 'rating_size':
            size_layer = Dense(1, activation='linear', name='predictions_size')(self.base)
            pred_layer = Dense(9, activation='linear', name='predictions')(self.base)
            outputs.append(pred_layer)
            outputs.append(size_layer)
        elif objective == 'distance-matrix':
            embed = Lambda(lambda x: x, name='embed_output')(self.base)
            outputs.append(embed)
        else:
            print("ERR: Illegual objective given ({})".format(objective))
            assert(False)

        if regularization_loss:
            embed = Lambda(lambda x: x, name='embed_output')(self.base)
            outputs.append(embed)

        self.model = Model(self.img_input, outputs, name='directArch')
        if regularization_loss:
            embed = Lambda(lambda x: x, name='embed_output')(x)
            outputs.append(embed)

        self.model = Model(self.img_input, outputs, name='directArch')

        self.input_shape = input_shape
        self.data_ready  = False
        self.model_ready = False

    def compile(self, learning_rate = 0.001, decay=0.1, loss='categorical_crossentropy', scheduale=[]):
        categorical_accuracy.__name__ = 'accuracy'
        sensitivity.__name__ = 'recall'
        root_mean_squared_error.__name__ = 'rmse'
        multitask_accuracy.__name__ = 'accuracy'
        metrics = []
        if self.objective == 'malignancy':
            metrics = [categorical_accuracy, f1, sensitivity, precision, specificity]
        elif self.objective == 'rating':
            metrics = {'predictions': root_mean_squared_error}
            loss = {'predictions': loss}
        elif self.objective == 'size':
            metrics = {'predictions_size': root_mean_squared_error}
            loss = {'predictions_size': loss}
        elif self.objective == 'rating_size':
            metrics = {'predictions': root_mean_squared_error,
                       'predictions_size': root_mean_squared_error}
            loss = {'predictions': loss,
                    'predictions_size': loss}
        elif self.objective == 'distance-matrix':
            loss = {'embed_output': loss}
            #metrics = {'embed_output': root_mean_squared_error}
        self.compile_model( lr=learning_rate, lr_decay=decay,
                            loss       = loss,  # 'categorical_crossentropy', mean_squared_error
                            metrics     = metrics,
                            scheduale=scheduale)
        self.lr         = learning_rate
        self.lr_decay   = decay
        self.model_ready = True

    def set_callbacks(self, label='', gen=False, do_graph=False):
        check = 'accuracy' if self.objective == 'malignancy' else 'loss'
        weight_file_pattern = self.set_weight_file_pattern(check, label)
        callbacks = self.callbacks
        callbacks += [ModelCheckpoint(weight_file_pattern, monitor='val_loss', save_best_only=False, save_weights_only=True)]
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
            no_pool_model.layers.pop()  # remove dense
            no_pool_model.layers.pop()  # remove normal
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
