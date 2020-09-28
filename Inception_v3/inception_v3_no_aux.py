#!/usr/bin/env python
# coding: utf-8

# inception_v3_no_aux.py

"""
Inception v3 model for Keras. 

It is the pure model of Inception v3. Since the Inception creators basically built the model on 
the linear algebra, incuding matrix components for inception a,b,c and reduction a,b. With regard
to inception stem, it is just addition computation. So the model is quite simple in the essence. 
The difficutly beghind the algebra is mainly the concept complxity. Please remember even the linear 
algebra includes huge gradients computing. 

Mr.Francois Cholett adds 5x5 kernel_size into his Inception v3 weights for downloading. However, 
the official Inception 3 paper is distinguished with 3x3 kernel_size in Inception A after excluding 
5x5 kernel_size. Therefore, the realization of script complies with the principle with adoption of 
3x3 kernel_size. Due to the reason, users need to generate the Inception 3 weights for a further
usage. If users want to run the model, please run the the command as follows.

$ python inception_v3_no_aux.py

After removing the auxilary layers, the total parameter of the Inception v3 has 19+ million. Please 
see the paper with opening the weblink as follows. Make the the necessary changes to adapt to the 
environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 11.0, cuDNN 8.0.1 and CUDA 450.57. In 
addition, write the new lines of code to replace the deprecated code. For the model summary, the 
size is set as 299 x 299 x 3. 

# Reference
- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)
"""

import tensorflow as tf
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Input, Conv2D, Dropout, Dense, Flatten, Activation, \
    BatchNormalization, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


WEIGHTS_PATH = '/home/mike/keras_dnn_models/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'home/mike/keras_dnn_models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


def conv_bn(x, filters, kernel_size, padding='same', strides=(1,1), name=None):
   
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, momentum=0.9998, scale=False)(x)
    x = Activation('relu')(x)

    return x


def inception_stem(input):

    x = conv_bn(input, filters=32, kernel_size=(3,3), strides=(2,2), padding='valid')
    x = conv_bn(x, filters=32, kernel_size=(3,3), padding='valid')
    x = conv_bn(x, filters=64, kernel_size=(3,3))
    x = MaxPooling2D(pool_size=(3,3), padding='same', strides=(2,2))(x)
    x = conv_bn(x, filters=80, kernel_size=(1,1), padding='valid')
    x = conv_bn(x, filters=192, kernel_size=(3,3), padding='valid')
    x = MaxPooling2D(pool_size=(3,3), padding='same', strides=(2,2))(x)

    return x


def inception_a(input):

    branch_11 = conv_bn(input, filters=64, kernel_size=(1,1), strides=(1,1))

    branch_12 = conv_bn(input, filters=48, kernel_size=(1,1), strides=(1,1) )
    branch_22 = conv_bn(branch_12, filters=64, kernel_size=(3,3), strides=(1,1))

    branch_13 = conv_bn(input, filters=64, kernel_size=(1,1), strides=(1,1))
    branch_23 = conv_bn(branch_13, filters=96, kernel_size=(3,3), strides=(1,1))
    branch_33 = conv_bn(branch_23, filters=96, kernel_size=(3,3), strides=(1,1))

    branch_14 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_24 = conv_bn(branch_14, filters=32, kernel_size=(1,1), strides=(1,1))

    x = concatenate([branch_11, branch_22, branch_33, branch_24], axis=3)
    
    return x


def reduction_a(input):

    branch_11 = conv_bn(input, filters=384, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_12 = conv_bn(input, filters=64, kernel_size=(1,1), strides=(1,1))
    branch_22 = conv_bn(branch_12, filters=96, kernel_size=(3,3), strides=(1,1))
    branch_32 = conv_bn(branch_22, filters=96, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_13 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input)

    x = concatenate([branch_11, branch_32, branch_13], axis=3)

    return x


def inception_b(input):

    branch_11 = conv_bn(input, filters=192, kernel_size=(1,1), strides=(1,1))

    branch_12 = conv_bn(input, filters=128, kernel_size=(1,1), strides=(1,1))
    branch_22 = conv_bn(branch_12, filters=128, kernel_size=(1,7), strides=(1,1))
    branch_32 = conv_bn(branch_22, filters=192, kernel_size=(7,1), strides=(1,1))

    branch_13 = conv_bn(input, filters=128, kernel_size=(1,1), strides=(1,1))
    branch_23 = conv_bn(branch_13, filters=128, kernel_size=(7,1), strides=(1,1))
    branch_33 = conv_bn(branch_23, filters=128, kernel_size=(1,7), strides=(1,1))
    branch_43 = conv_bn(branch_33, filters=128, kernel_size=(7,1), strides=(1,1))
    branch_53 = conv_bn(branch_43, filters=192, kernel_size=(1,7), strides=(1,1))

    branch_14 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_24 = conv_bn(branch_14, filters=192, kernel_size=(1,1), strides=(1,1))

    x = concatenate([branch_11, branch_32, branch_53, branch_24], axis=3)

    return x  


def reduction_b(input):

    branch_11 = conv_bn(input, filters=192, kernel_size=(1,1), strides=(1,1))
    branch_21 = conv_bn(branch_11, filters=320, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_12 = conv_bn(input, filters=192, kernel_size=(1,1), strides=(1,1))
    branch_22 = conv_bn(branch_12, filters=192, kernel_size=(1,7), strides=(1,1))
    branch_32 = conv_bn(branch_22, filters=192, kernel_size=(7,1), strides=(1,1))
    branch_42 = conv_bn(branch_32, filters=192, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_13 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input)

    x = concatenate([branch_21, branch_42, branch_13], axis=3)

    return x


def inception_c(input):

    branch_11 = conv_bn(input, filters=320, kernel_size=(1,1), strides=(1,1))

    branch_12 = conv_bn(input, filters=384, kernel_size=(1,1), strides=(1,1))
    branch_22 = conv_bn(branch_12, filters=384, kernel_size=(1,3), strides=(1,1))
    branch_23 = conv_bn(branch_12, filters=384, kernel_size=(3,1), strides=(1,1))
    branch_33 = concatenate([branch_22, branch_23], axis=3) # sub-concatenate

    branch_14 = conv_bn(input, filters=448, kernel_size=(1,1), strides=(1,1))
    branch_24 = conv_bn(branch_14, filters=384, kernel_size=(3,3), strides=(1,1))
    branch_34 = conv_bn(branch_24, filters=384, kernel_size=(1,3), strides=(1,1))
    branch_35 = conv_bn(branch_24, filters=384, kernel_size=(3,1), strides=(1,1))
    branch_45 = concatenate([branch_34,branch_35], axis=3) # sub-concatenate

    branch_16 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_26 = conv_bn(branch_16, filters=192, kernel_size=(1,1), strides=(1,1))

    # concatenate the four parts
    x = concatenate([branch_11, branch_33, branch_45, branch_26], axis=3) 

    return x

"""
# It is non-auxilary version, so we commence Auxilary classifier 
def aux_output(x):
    y = AveragePooling2D(pool_size=(5,5), strides=(3,3), padding='same')(x)
    y = Conv2D(filters=128, kernel_size=(1,1), padding='same', activation='relu')(y)
    y = Dense(units=1024, activation='relu')(y)
    y = Dropout(0.7)(y)
    y = Dense(units=num_classes, activation='sigmoid')(y)
    return y
"""

def inception_v3(input_shape, num_classes, weights=None, include_top=None):
    # Build the abstract Inception v4 network
    """
    Args:
        input_shape: three dimensions in the TensorFlow Data Format
        num_classes: number of classes
        weights: pre-defined Inception v3 weights with ImageNet
        include_top: a boolean, for full traning or finetune 
    Return: 
        logits: the logit outputs of the model.
    """
    # Initizate a 3D shape(weight,height,channels) into a 4D tensor(batch, 
    # weight,height,channels). If no batch size, it is defaulted as None.
    inputs = Input(shape=input_shape)

    x = inception_stem(inputs)

    # 3 x Inception-A blocks
    for i in range(0, 3):
        x = inception_a(x)  

    # Reduction-A block
    x = reduction_a(x)

    # 4 x Inception-B blocks
    for i in range(0, 4):
        x = inception_b(x)

    # Commence the auxiliary classifier 
    # -y = aux_output(x)

    # Reduction-B block
    x = reduction_b(x)

    # 2 x Inception-C blocks
    for i in range(0, 2):
        x = inception_c(x)

    # It comply with the Inception v3 paper, so it is different from Cholett's snippet. 
    if include_top:
        x = AveragePooling2D((8,8), padding='valid')(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(units=num_classes, activation='softmax')(x)

    # Only keep x in the 4D tensor if no auxilary classifier 
    model = Model(inputs, x, name='inception_v3')
    # model = Model(inputs, [x,y], name='inception_v3')

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
    include_top=True

    model = inception_v3(input_shape, num_classes, include_top)

    model.summary()
