#!/usr/bin/env python
# coding: utf-8

# inception_v4_prematrix.py

"""
Inception V4 model for Keras. 

If the files of nception_v4_weights are hard to download during the runtime due to the hindrince 
of network connectivity, users can directly download then and then and then run the script. 

$ python inception_v4_pred.py

Inception Weights 
https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels.h5
Inception Weights with No Top
https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5'

Make the the necessary changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 
11.0, cuDNN 8.0.1 and CUDA 450.57. In addition, write the new lines of code to replace the deprecated 
code. 

Francios Chollet - inception_v3.py 
Kent Sommers - inception_v4.py
Note that the input image format for this model is different than for the VGG16 and ResNet models 
(299x299 instead of 224x224), and that the input preprocessing function is also different (same as 
Xception).

http://arxiv.org/pdf/1602.07261v1.pd 
"""

import numpy as np
import tensorflow as tf
import warnings

from keras.layers.convolutional import MaxPooling2D, Conv2D, AveragePooling2D
from keras.layers import Input, Dropout, Dense, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate

# -from keras import regularizers, initializers
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


# Assume that users have already download the Inception v4 weights 
WEIGHTS_PATH = '/home/mike/keras_dnn_models/inception-v4_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = '/home/mike/keras_dnn_models/inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5'


# Define the joint function to apply conv + BN. 
def conv2d_bn(x, filters, kernel_size, strides, padding='same', use_bias=False):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
               kernel_initializer="he_normal", kernel_regularizer=l2(0.00004))(x)
    x = BatchNormalization(axis=bn_axis, momentum=0.9997, scale=False)(x)
    x = Activation('relu')(x)

    return x


def block_inception_a(input):

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    branch_11 = conv2d_bn(input, filters=96, kernel_size=(1,1), strides=(1,1))

    branch_12 = conv2d_bn(input, filters=64, kernel_size=(1,1), strides=(1,1) )
    branch_22 = conv2d_bn(branch_12, filters=96, kernel_size=(3,3), strides=(1,1))

    branch_13 = conv2d_bn(input, filters=64, kernel_size=(1,1), strides=(1,1))
    branch_23 = conv2d_bn(branch_13, filters=96, kernel_size=(3,3), strides=(1,1))
    branch_33 = conv2d_bn(branch_23, filters=96, kernel_size=(3,3), strides=(1,1))

    branch_14 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_24 = conv2d_bn(branch_14, filters=96, kernel_size=(1,1), strides=(1,1))

    x = concatenate([branch_11,branch_22,branch_33,branch_24], axis=bn_axis)
    
    return x


def block_reduction_a(input):

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    branch_11 = conv2d_bn(input, filters=384, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_12 = conv2d_bn(input, filters=192, kernel_size=(1,1), strides=(1,1))
    branch_22 = conv2d_bn(branch_12, filters=224, kernel_size=(3,3), strides=(1,1))
    branch_32 = conv2d_bn(branch_22, filters=256, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_13 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input)

    x = concatenate([branch_11,branch_32,branch_13], axis=bn_axis)

    return x


def block_inception_b(input):

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    branch_11 = conv2d_bn(input, filters=384, kernel_size=(1,1), strides=(1,1))

    branch_12 = conv2d_bn(input, filters=192, kernel_size=(1,1), strides=(1,1))
    branch_22 = conv2d_bn(branch_12, filters=224, kernel_size=(1,7), strides=(1,1))
    branch_32 = conv2d_bn(branch_22, filters=256, kernel_size=(7,1), strides=(1,1))

    branch_13 = conv2d_bn(input, filters=192, kernel_size=(1,1), strides=(1,1))
    branch_23 = conv2d_bn(branch_13, filters=192, kernel_size=(7,1), strides=(1,1))
    branch_33 = conv2d_bn(branch_23, filters=224, kernel_size=(1,7), strides=(1,1))
    branch_43 = conv2d_bn(branch_33, filters=224, kernel_size=(7,1), strides=(1,1))
    branch_53 = conv2d_bn(branch_43, filters=256, kernel_size=(1,7), strides=(1,1))

    branch_14 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_24 = conv2d_bn(branch_14, filters=128, kernel_size=(1,1), strides=(1,1))

    x = concatenate([branch_11,branch_32,branch_53,branch_24], axis=bn_axis)

    return x


def block_reduction_b(input):

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    branch_11 = conv2d_bn(input, filters=192, kernel_size=(1,1), strides=(1,1))
    branch_21 = conv2d_bn(branch_11, filters=192, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_12 = conv2d_bn(input, filters=256, kernel_size=(1,1), strides=(1,1))
    branch_22 = conv2d_bn(branch_12, filters=256, kernel_size=(1,7), strides=(1,1))
    branch_32 = conv2d_bn(branch_22, filters=320, kernel_size=(7,1), strides=(1,1))
    branch_42 = conv2d_bn(branch_32, filters=320, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_13 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input)

    x = concatenate([branch_21,branch_42,branch_13], axis=bn_axis)

    return x


def block_inception_c(input):

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    branch_11 = conv2d_bn(input, filters=256, kernel_size=(1,1), strides=(1,1))

    branch_12 = conv2d_bn(input, filters=384, kernel_size=(1,1), strides=(1,1))
    branch_22 = conv2d_bn(branch_12, filters=256, kernel_size=(1,3), strides=(1,1))
    branch_23 = conv2d_bn(branch_12, filters=256, kernel_size=(3,1), strides=(1,1))
    branch_33 = concatenate([branch_22,branch_23], axis=bn_axis)

    branch_14 = conv2d_bn(input, filters=384, kernel_size=(1,1), strides=(1,1))
    branch_24 = conv2d_bn(branch_14, filters=448, kernel_size=(3,1), strides=(1,1))
    branch_34 = conv2d_bn(branch_24, filters=512, kernel_size=(1,3), strides=(1,1))
    branch_44 = conv2d_bn(branch_34, filters=256, kernel_size=(1,3), strides=(1,1))
    branch_45 = conv2d_bn(branch_34, filters=256, kernel_size=(3,1), strides=(1,1))
    branch_55 = concatenate([branch_44,branch_45], axis=bn_axis)

    branch_16 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_26 = conv2d_bn(branch_16, filters=256, kernel_size=(1,1), strides=(1,1))

    x = concatenate([branch_11,branch_33,branch_55,branch_26], axis=bn_axis)

    return x


def inception_v4_base(input):

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    basenet = conv2d_bn(input, filters=32, kernel_size=(3,3), strides=(2,2), padding='valid')
    basenet = conv2d_bn(basenet, filters=32, kernel_size=(3,3), strides=(1,1), padding='valid')
    basenet = conv2d_bn(basenet, filters=64, kernel_size=(3,3), strides=(1,1))

    branch_11 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(basenet)
    branch_12 = conv2d_bn(basenet, filters=96, kernel_size=(3,3), strides=(2,2), padding='valid')
    basenet = concatenate([branch_11, branch_12], axis=bn_axis)

    branch_13 = conv2d_bn(basenet, filters=64, kernel_size=(1,1), strides=(1,1))
    branch_14 = conv2d_bn(branch_13, filters=96, kernel_size=(3,3), strides=(1,1), padding='valid')

    branch_15 = conv2d_bn(basenet, filters=64, kernel_size=(1,1), strides=(1,1))
    branch_16 = conv2d_bn(branch_15, filters=64, kernel_size=(1,7), strides=(1,1))
    branch_17 = conv2d_bn(branch_16, filters=64, kernel_size=(7,1), strides=(1,1))
    branch_18 = conv2d_bn(branch_17, filters=96, kernel_size=(3,3),  strides=(1,1), padding='valid')

    basenet = concatenate([branch_14,branch_18], axis=bn_axis)

    branch_19 = conv2d_bn(basenet, filters=192, kernel_size=(3,3), strides=(2,2), padding='valid')
    branch_20 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(basenet)

    basenet = concatenate([branch_19,branch_20], axis=bn_axis)
  
    # 4 x Inception-A blocks: 35 x 35 x 384
    for idx in range(4):
        basenet = block_inception_a(basenet)

    # Reduction-A block: # 35 x 35 x 384
    basenet = block_reduction_a(basenet)

    # 7 x Inception-B blocks: 17 x 17 x 1024
    for idx in range(7):
        basenet = block_inception_b(basenet)

    # Reduction-B block: 17 x 17 x 1024
    basenet = block_reduction_b(basenet)

    # 3 x Inception-C blocks: 8 x 8 x 1536
    for idx in range(3):
        basenet = block_inception_c(basenet)

    return basenet


def inception_v4(num_classes, dropout_prob, weights, include_top):
    '''
    Creates the inception v4 network
    Args:
        num_classes: number of classes
        dropout_prob: float, the fraction to keep before final layer.
    
    Returns: 
        logits: the logits outputs of the model.
    '''

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    if K.image_data_format() == 'channels_last':
        inputs = Input((299, 299, 3))
    else:
        inputs = Input((3, 299, 299))

    # Make the inception base
    x = inception_v4_base(inputs)


    # Final pooling and prediction
    if include_top:
        # 1 x 1 x 1536
        x = AveragePooling2D((8,8), padding='valid')(x)
        x = Dropout(dropout_prob)(x)
        x = Flatten()(x)
        # 1536
        x = Dense(units=num_classes, activation='softmax')(x)

    model = Model(inputs, x, name='inception_v4')

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_last':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend'
                              'with image data format convention '
                              '(`image_data_format="channels_last"`).'
                              'If you meet a problem, please check the '
                              'Keras config at ~/.keras/keras.json.')
        if include_top:
            weights_path = WEIGHTS_PATH
        else:
            weights_path = WEIGHTS_PATH_NO_TOP
        # -model.load_weights(weights_path, by_name=True)
        model.load_weights(weights_path)

    return model


def preprocess_input(x):

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    output = np.multiply(x, 2.0)

    return output


if __name__ == '__main__':

    num_classes = 1001
    dropout_prob = 0.2 
    weights='imagenet'
    include_top = True 

    model = inception_v4(num_classes, dropout_prob, weights, include_top)
    model.summary()

    img_path = '/home/mike/Documents/keras_inception_v4/elephant.jpg'
    img = image.load_img(img_path, target_size=(299,299))
    output = preprocess_input(img)

    # Open the class label dictionary(that is a human readable label-given ID)
    classes = eval(open('/home/mike/Documents/keras_inception_v4/validation_utils/class_names.txt', 'r').read())

    # Run the prediction on the given image
    preds = model.predict(output)

    # Delete Keras's decode_predictions() that accepts 1000 rather than 1001 classes defaulted in Inception v4 weights
    # -print('Predicted:', decode_predictions(preds))
    print("Class is: " + classes[np.argmax(preds)-1])
    print("Certainty is: " + str(preds[0][np.argmax(preds)]))