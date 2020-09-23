#!/usr/bin/env python
# coding: utf-8

# inception_v4_cbase.py

"""
Inception V4 model for Keras. 

If the files of Inception_v4_weights are hard to download during the runtime due to the hindrince 
of network connectivity, users can directly download them. There is a note for the download link
in the file of Inception_v4_predict. 

It is the pure convbase(convolutional base) backend that is called by any client include dense layer 
functionality. The seperation is useful for a flexible call. 

Please notify that following details hided in the reliazation. 

We set "inputs" as actual arguments in the function of inceptin_v4() and "input" as formal arguments
in all the functions except conv_bn(). User can change "inputs" to "input". It can product the correct 
classification. However, the meanings are different, i.e., they are involved in the functional call.   

# inputs and input 

The line of code is extremely important. It is placed after the BatchNorm rather than being placed
in the Conv2D(function). In the contrary, the classification would be grealy diminishing. 

# x = Activation('relu')(x) 

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

from keras.layers import Conv2D, Activation, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate

from keras.initializers import he_normal
from keras.regularizers import l2


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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