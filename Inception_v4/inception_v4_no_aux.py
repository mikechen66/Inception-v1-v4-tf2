#!/usr/bin/env python
# coding: utf-8

# inception_v4_no_aux.py

"""
Inception V4 model for Keras. 

If the files of Inception_v4_weights are hard to download during the runtime due to the hindrince 
of network connectivity, users can directly download them. There is a note for the download link
in the file of Inception_v4_predict. 

Inception v4 Weights 
https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels.h5

Inception v4 Weights with No Top
https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5'

After downloading the weights and adding them into the script, users can run the script with the 
command as follows. 

$ python inception_v4_no_aux.py

Make the the necessary changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 
11.0, cuDNN 8.0.1 and CUDA 450.57. In addition, write the new lines of code to replace the deprecated 
code. 

The script has many changes on the foundation of Inception v3 by Francios Chollet, Inception v4 by 
Kent Sommers and many other published results. I would like to thank all of them for the contributions. 

Note that the input image format for this model is different than for the VGG16 and ResNet models 
(299x299 instead of 224x224), and that the input preprocessing function is also different (same as 
Xception).
http://arxiv.org/pdf/1602.07261v1.pd 
"""

import numpy as np
import tensorflow as tf
import warnings

from keras.layers import Input, Conv2D, Dropout, Dense, Flatten, Activation, BatchNormalization, \
    MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate

from keras.initializers import he_normal
from keras.regularizers import l2
from keras.models import Model

from keras import backend as K
# -from keras.applications.imagenet_utils import decode_predictions
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.preprocessing import image


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# The defualted weights are suitable to the model because it exlcue the auxilary classifier
WEIGHTS_PATH = '/home/mike/keras_dnn_models/inception-v4_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = '/home/mike/keras_dnn_models/inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5'


def conv_bn(x, filters, kernel_size, strides, padding='same', use_bias=False):
    # Define the joint function to apply the combination of Conv and BatchNorm 
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
               kernel_initializer="he_normal", kernel_regularizer=l2(0.00004))(x)
    x = BatchNormalization(axis=3, momentum=0.9997, scale=False)(x)
    x = Activation('relu')(x)

    return x


def inception_stem(input):
    # Define the stem network 
    stem = conv_bn(input, filters=32, kernel_size=(3,3), strides=(2,2), padding='valid')
    stem = conv_bn(stem, filters=32, kernel_size=(3,3), strides=(1,1), padding='valid')
    stem = conv_bn(stem, filters=64, kernel_size=(3,3), strides=(1,1))

    branch_11 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(stem)
    branch_12 = conv_bn(stem, filters=96, kernel_size=(3,3), strides=(2,2), padding='valid')
    
    stem = concatenate([branch_11, branch_12], axis=3)

    branch_13 = conv_bn(stem, filters=64, kernel_size=(1,1), strides=(1,1))
    branch_14 = conv_bn(branch_13, filters=96, kernel_size=(3,3), strides=(1,1), padding='valid')
    branch_15 = conv_bn(stem, filters=64, kernel_size=(1,1), strides=(1,1))
    branch_16 = conv_bn(branch_15, filters=64, kernel_size=(1,7), strides=(1,1))
    branch_17 = conv_bn(branch_16, filters=64, kernel_size=(7,1), strides=(1,1))
    branch_18 = conv_bn(branch_17, filters=96, kernel_size=(3,3),  strides=(1,1), padding='valid')
    
    stem = concatenate([branch_14,branch_18], axis=3)

    branch_19 = conv_bn(stem, filters=192, kernel_size=(3,3), strides=(2,2), padding='valid')
    branch_20 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(stem)
    
    x = concatenate([branch_19,branch_20], axis=3)
  
    return x


def inception_a(input):

    branch_11 = conv_bn(input, filters=96, kernel_size=(1,1), strides=(1,1))

    branch_12 = conv_bn(input, filters=64, kernel_size=(1,1), strides=(1,1) )
    branch_22 = conv_bn(branch_12, filters=96, kernel_size=(3,3), strides=(1,1))

    branch_13 = conv_bn(input, filters=64, kernel_size=(1,1), strides=(1,1))
    branch_23 = conv_bn(branch_13, filters=96, kernel_size=(3,3), strides=(1,1))
    branch_33 = conv_bn(branch_23, filters=96, kernel_size=(3,3), strides=(1,1))

    branch_14 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_24 = conv_bn(branch_14, filters=96, kernel_size=(1,1), strides=(1,1))

    x = concatenate([branch_11,branch_22,branch_33,branch_24], axis=3)
    
    return x


def reduction_a(input):

    branch_11 = conv_bn(input, filters=384, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_12 = conv_bn(input, filters=192, kernel_size=(1,1), strides=(1,1))
    branch_22 = conv_bn(branch_12, filters=224, kernel_size=(3,3), strides=(1,1))
    branch_32 = conv_bn(branch_22, filters=256, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_13 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input)

    x = concatenate([branch_11,branch_32,branch_13], axis=3)

    return x


def inception_b(input):

    branch_11 = conv_bn(input, filters=384, kernel_size=(1,1), strides=(1,1))

    branch_12 = conv_bn(input, filters=192, kernel_size=(1,1), strides=(1,1))
    branch_22 = conv_bn(branch_12, filters=224, kernel_size=(1,7), strides=(1,1))
    branch_32 = conv_bn(branch_22, filters=256, kernel_size=(7,1), strides=(1,1))

    branch_13 = conv_bn(input, filters=192, kernel_size=(1,1), strides=(1,1))
    branch_23 = conv_bn(branch_13, filters=192, kernel_size=(7,1), strides=(1,1))
    branch_33 = conv_bn(branch_23, filters=224, kernel_size=(1,7), strides=(1,1))
    branch_43 = conv_bn(branch_33, filters=224, kernel_size=(7,1), strides=(1,1))
    branch_53 = conv_bn(branch_43, filters=256, kernel_size=(1,7), strides=(1,1))

    branch_14 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_24 = conv_bn(branch_14, filters=128, kernel_size=(1,1), strides=(1,1))

    x = concatenate([branch_11,branch_32,branch_53,branch_24], axis=3)

    return x


def reduction_b(input):

    branch_11 = conv_bn(input, filters=192, kernel_size=(1,1), strides=(1,1))
    branch_21 = conv_bn(branch_11, filters=192, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_12 = conv_bn(input, filters=256, kernel_size=(1,1), strides=(1,1))
    branch_22 = conv_bn(branch_12, filters=256, kernel_size=(1,7), strides=(1,1))
    branch_32 = conv_bn(branch_22, filters=320, kernel_size=(7,1), strides=(1,1))
    branch_42 = conv_bn(branch_32, filters=320, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_13 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input)

    x = concatenate([branch_21,branch_42,branch_13], axis=3)

    return x


def inception_c(input):

    branch_11 = conv_bn(input, filters=256, kernel_size=(1,1), strides=(1,1))

    branch_12 = conv_bn(input, filters=384, kernel_size=(1,1), strides=(1,1))
    branch_22 = conv_bn(branch_12, filters=256, kernel_size=(1,3), strides=(1,1))
    branch_23 = conv_bn(branch_12, filters=256, kernel_size=(3,1), strides=(1,1))
    branch_33 = concatenate([branch_22,branch_23], axis=3)

    branch_14 = conv_bn(input, filters=384, kernel_size=(1,1), strides=(1,1))
    branch_24 = conv_bn(branch_14, filters=448, kernel_size=(3,1), strides=(1,1))
    branch_34 = conv_bn(branch_24, filters=512, kernel_size=(1,3), strides=(1,1))
    branch_44 = conv_bn(branch_34, filters=256, kernel_size=(1,3), strides=(1,1))
    branch_45 = conv_bn(branch_34, filters=256, kernel_size=(3,1), strides=(1,1))
    branch_55 = concatenate([branch_44,branch_45], axis=3)

    branch_16 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_26 = conv_bn(branch_16, filters=256, kernel_size=(1,1), strides=(1,1))

    x = concatenate([branch_11,branch_33,branch_55,branch_26], axis=3)

    return x
    

"""
def aux_output(x):
# Define an auxilary classifier 
    y = AveragePooling2D(pool_size=(5,5), strides=(3,3), padding='same')(x)
    y = Conv2D(filters=128, kernel_size=(1,1), padding='same', activation='relu')(y)
    y = GlobalAveragePooling2D(name='aux_gav_pool')(x)
    y = Dense(units=768, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(units=num_classes, activation='softmax', name="aux_pred")(y)

    return y
"""


def inception_v4(input_shape, num_classes, include_top, weights=None):
    # Build the abstract Inception v4 network
    '''
    Args:
        input_shape: three dimensions in the TensorFlow Data Format
        num_classes: number of classes
        weights: pre-defined Inception v4 weights 
        include_top: a boolean, for full traning or finetune 
    Return: 
        logits: the logit outputs of the model.
    '''
    inputs = Input(shape=input_shape)

    # Make the the stem of Inception v4 
    x = inception_stem(inputs)

    # 4 x Inception-A blocks
    for i in range(0, 4):
        x = inception_a(x)

    # Reduction-A block
    x = reduction_a(x)

    # 7 x Inception-B blocks
    for i in range(0, 7):
        x = inception_b(x)

    # Please note it is the auxiliary classifier 
    # -y = aux_output(x)

    # Reduction-B block
    x = reduction_b(x)

    # 3 x Inception-C blocks
    for i in range(0, 3):
        x = inception_c(x)

    # Final pooling and prediction
    if include_top:
        x = GlobalAveragePooling2D(name='main_gav_pool')(x)
        x = Dense(units=1536, activation='relu')(x) 
        x = Dropout(0.5)(x)
        x = Dense(units=num_classes, activation='softmax', name='main_pred')(x)

    # -model = Model(inputs, [x,y], name='inception_v4')
    model = Model(inputs, x, name='inception_v4')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = WEIGHTS_PATH
        else:
            weights_path = WEIGHTS_PATH_NO_TOP
        # -model.load_weights(weights_path, by_name=True)
        model.load_weights(weights_path)

    return model


if __name__ == '__main__':

    input_shape = (299, 299, 3)
    num_classes = 1000
    include_top = True 
    weights='imagenet'

    # -model = inception_v4(input_shape, num_classes, weights, include_top)
    model = inception_v4(input_shape, num_classes, weights)
    
    model.summary()
