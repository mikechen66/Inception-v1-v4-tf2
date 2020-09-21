#!/usr/bin/env python
# coding: utf-8

# incepion_v2_func_l2.py

"""
The script is the function-style model of GoogLeNet Inception v2 with l2 regularization. The 
second-stage model is used for the thorough study on how Google has built the network based 
with BatchNorm on the network depth of AlexNet and the small filter size of NIN(Network In 
Network). It stacks the layers to address the complexity of the image classification. Please 
use the following command to run the script. 

# $ python inception_v2_func_ls.py

It is quite strange it is hard to get the code of Inception v2 in either Google Search or Github. 
Futhermotre, most of the available Inception v2 has more than total size of 20+ million (or even 
200 mllion) parameters in the condition of 1000 classes and input size of (224,224,3). That is 
amazing. In contrast, the official Inception v2 has lass than 10 million. Therefore, the new-built
script is deidicated to provide users with the accurate model incarnation with the total size of 
parameters to 8+ million. It is quite close to the official published parameter size by GoogleNet 
Inception Team. 

Notes:

Please remmeber that it is the TensorFlow realization with image_data_foramt = 'channels_last'. If
the env of Keras is 'channels_first', please change it according to the TensorFlow convention. 

The realizatin is eight inception layers after deducting the inception_4e. In contrast, the script 
of Inception v1 includes the layer of inception_4e. If users want more layers, it is necessary to 
comply with the prerequize of the following paper of Inception. 

Environment: 

Ubuntu 18.04 
TensorFlow 2.3
Keras 2.4.3
CUDA Toolkit 11.0 
cuDNN 8.0.1
CUDA 450.57 

Reference: 
Rethinking the Inception Architecture for Computer Vision(GoogLeNet Inception v2)
https://arxiv.org/pdf/1512.00567v3.pdf
"""


import keras
import numpy as np

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, \
    Flatten, Dense, Dropout, BatchNormalization, Activation, Input, concatenate
from keras.models import Model
from keras.initializers import he_normal
from keras.regularizers import l2
    

# Create the model 
def googlenet(input_shape, num_classes):

    input = Input(shape=input_shape)

    x = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu', kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(input)       
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    
    x = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    
    x = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)  
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    
    x = inception(x, axis=3, params=[(64,),(64,64),(64,96,96),(32,)]) 
    x = inception(x, axis=3, params=[(64,),(64,64),(64,96,96),(64,)]) 
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

    x = inception(x, axis=3, params=[(224,),(64,96),(96,128,128),(128,)]) 
    x = inception(x, axis=3, params=[(192,),(96,128),(96,128,128),(128,)]) 
    x = inception(x, axis=3, params=[(160,),(128,160),(128,160,160),(96,)]) 
    x = inception(x, axis=3, params=[(96,),(128,192),(160,192,192),(96,)]) 
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

    x = inception(x, axis=3, params=[(352,),(192,320),(160,224,224),(128,)]) 
    x = inception(x, axis=3, params=[(352,),(192,320),(192,224,224),(128,)]) 
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)


    # Add the average pooling 
    x = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(x)
    x = Dropout(rate=0.4)(x)
    linear = Dense(num_classes, activation='softmax', kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)

    model = Model(inputs=input, outputs=linear)          

    return model


# Define the inception v2 module 
def inception(x, params, axis):
    
    # Bind the vertical cells tegother for an elegant realization
    [branch1, branch2, branch3, branch4] = params

    conv11 = Conv2D(filters=branch1[0], kernel_size=(1,1), strides=1, padding='same', activation='relu', kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)
    bn11 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv11)

    conv12 = Conv2D(filters=branch2[0], kernel_size=(1,1), strides=1, padding='same', activation='relu', kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)
    bn12 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv12)
    
    conv22 = Conv2D(filters=branch2[1], kernel_size=(3,3), strides=1, padding='same', activation='relu', kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(bn12)
    bn22 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv22)

    conv13 = Conv2D(filters=branch3[0], kernel_size=(1,1), strides=1, padding='same', activation='relu', kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)
    bn13 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv13)
    conv23 = Conv2D(filters=branch3[1], kernel_size=(3,3), strides=1, padding='same', activation='relu', kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(bn13)
    bn23 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv23)
    conv33 = Conv2D(filters=branch3[2], kernel_size=(3,3), strides=1, padding='same', activation='relu', kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(bn23)
    bn33 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv33)

    mp14 = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)
    conv24 = Conv2D(filters=branch4[0], kernel_size=(1,1), strides=1, padding='same', activation='relu', kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(mp14)
    bn24 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv24)

    inception_output = concatenate([bn11,bn22,bn33,bn24], axis=3)

    return inception_output


if __name__ == '__main__':

    num_classes = 1000
    image_width = 224
    image_height = 224
    channels = 3

    # Assign the values 
    input_shape = (image_width, image_height, channels)

    model = googlenet(input_shape, num_classes)

    model.summary()
