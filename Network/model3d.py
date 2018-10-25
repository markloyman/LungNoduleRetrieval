# -*- coding: utf-8 -*-
'''
Core Embedding Network

Based on:
[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
'''
from __future__ import absolute_import
from __future__ import print_function

import warnings

from keras import backend as K
from keras import layers
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D, Dense
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.layers import Lambda, Bidirectional, Masking
from keras.layers import Flatten
from keras.layers import ActivityRegularization
from keras.models import Model


def gru3d_loader(input_tensor=None, input_shape=None, weights=None, output_size=1024, return_model=False,
                        pooling=None, normalize=False, binary=False, regularize=None):

    if K.backend() != 'tensorflow':
        raise RuntimeError('The model is only available with '
                           'the TensorFlow backend.')
    if K.image_data_format() != 'channels_last':
        warnings.warn('The  model is only available for the '
                      'input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height). '
                      'You should set `image_data_format="channels_last"` in your Keras '
                      'config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    # Determine proper input shape
    '''
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=299,
                                      min_size=71,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)
    '''
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    merge = 'sum'
    x = img_input  # Flatten()(img_input)
    x = Masking(mask_value=0.0)(x)
    x = Bidirectional(GRU(units=output_size, dropout=0.4, recurrent_dropout=0.6, return_sequences=True), merge_mode=merge)(x)
    x = Bidirectional(GRU(units=output_size, dropout=0.2, recurrent_dropout=0.4, return_sequences=True), merge_mode=merge)(x)
    #x = Bidirectional(GRU(units=output_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), merge_mode=merge)(x)
    #x = Bidirectional(GRU(units=output_size, dropout=0, recurrent_dropout=0, return_sequences=True), merge_mode=merge)(x)
    x = Bidirectional(GRU(units=output_size, dropout=0.0, recurrent_dropout=0.2, return_sequences=False), merge_mode=merge)(x)

    #x = Dense(units=output_size)(x)
    #x = Dense(units=output_size)(x)

    '''
    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='embeding')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='embeding')(x)
    elif pooling == 'rmac':
        # we have x16 reduction, so 128*128 input was reduced to 8*8
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='embed_pool')(x)
        x = GlobalAveragePooling2D(name='embeding')(x)
    elif pooling == 'msrmac':
        s1 = GlobalAveragePooling2D(name='s1')(x)
        s2 = MaxPooling2D((2, 2), strides=(1, 1), padding='valid')(x)
        s2 = GlobalAveragePooling2D(name='s2')(s2)
        s4 = MaxPooling2D((4, 4), strides=(2, 2), padding='valid')(x)
        s4 = GlobalAveragePooling2D(name='s4')(s4)
        s8 = GlobalMaxPooling2D(name='s8')(x)
        x = layers.add([s1, s2, s4, s8], name='embeding')
    elif pooling == 'conv':
        x = Conv2D(output_size, (8, 8), strides=(1, 1), use_bias=False, name='embed_conv')(x)
        x = Flatten(name='embeding')(x)
    else:
        x = Flatten(name='embeding')(x)

    '''

    if binary:
        x = Activation('sigmoid')(x)

    if normalize:
        x = Lambda(lambda q: K.l2_normalize(q, axis=-1), name='n_embedding')(x)

    if regularize is not None:
        #x = Activation('relu', name='prereg_act')(x)
        x = ActivityRegularization(l1=regularize)(x)

    # load weights
    if weights is not None:
        x.load_weights('Weights/' + weights)
        print('Loaded {}'.format(weights))

    if old_data_format:
        K.set_image_data_format(old_data_format)

    if return_model:
        return Model(img_input, x)
    else:
        return x