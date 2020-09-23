#!/usr/bin/env python
# coding: utf-8

# incepion_v1_func.py

"""
The script is the function-style sequentual model of GoogLeNet Inception v1. The first-stage model 
is used for the thorough study on how Google has built the network based on the network depth of 
AlexNet and the small filter size of NIN(Network In Network). It stacks the layers to address the 
complexity of the image classification. Please use the following command to run the script. 

$ python inception_v1_func.py

It is quite strange that most of the available Inception v1 has more than total size of 10+ million 
parameters. In contrast, the official Inception v1 has only the 5+ million. Therefore, the modified 
script downsizes the total size of parameters to 6+ million. It is close to the official published
parameter size. 

Environment: 

Ubuntu 18.04 
TensorFlow 2.3
Keras 2.4.3
CUDA Toolkit 11.0, 
cuDNN 8.0.1
CUDA 450.57. 

Reference: 
Going Deeper with Convolutions(GoogLeNet Inception v1)
https://arxiv.org/pdf/1409.4842.pdf
"""


from keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras.layers.merge import concatenate
from keras.regularizers import l2


def googlenet(input_shape, num_classes):

    input = Input(shape=input_shape)

    conv1_7x7 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu', 
                       kernel_regularizer=l2(0.01))(input)
    maxpool1_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv1_7x7)
    conv2_1x1 = Conv2D(filters=64, kernel_size=(1,1),  strides=(1,1), padding='same', activation='relu', 
                       kernel_regularizer=l2(0.01))(maxpool1_3x3)
    conv2_3x3 = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', 
                       kernel_regularizer=l2(0.01))(conv2_1x1)
    maxpool2_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv2_3x3)

    inception_3a = inception(input=maxpool2_3x3, axis=3, params=[(64,),(96,128),(16,32),(32,)])
    inception_3b = inception(input=inception_3a, axis=3, params=[(128,),(128,192),(32,96),(64,)])
    maxpool3_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(inception_3b)

    inception_4a = inception(input=maxpool3_3x3, axis=3, params=[(192,),(96,208),(16,48),(64,)])
    inception_4b = inception(input=inception_4a, axis=3, params=[(160,),(112,224),(24,64),(64,)])
    inception_4c = inception(input=inception_4b, axis=3, params=[(128,),(128,256),(24,64),(64,)])
    inception_4d = inception(input=inception_4c, axis=3, params=[(112,),(144,288),(32,64),(64,)])
    inception_4e = inception(input=inception_4d, axis=3, params=[(256,),(160,320),(32,128),(128,)])
    maxpool4_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(inception_4e)

    inception_5a = inception(input=maxpool4_3x3, axis=3, params=[(256,),(160,320),(32,128),(128,)])
    inception_5b = inception(input=inception_5a, axis=3, params=[(384,),(192,384),(48,128),(128,)]) 
    avgpool1_7x7 = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(inception_5b)

    drop = Dropout(rate=0.4)(avgpool1_7x7)
    linear = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(drop)
    
    model = Model(inputs=input, outputs=linear)

    return model 


def inception(input, axis, params):

    # Bind the vertical cells tegother for an elegant realization 
    [branch1, branch2, branch3, branch4] = params

    conv_11 = Conv2D(filters=branch1[0], kernel_size=(1,1), padding='same', activation='relu', 
                     kernel_regularizer=l2(0.01))(input)

    conv_12 = Conv2D(filters=branch2[0], kernel_size=(1,1), padding='same', activation='relu', 
                     kernel_regularizer=l2(0.01))(input)
    conv_22 = Conv2D(filters=branch2[1], kernel_size=(3,3), padding='same', activation='relu', 
                     kernel_regularizer=l2(0.01))(conv_12)

    conv_13 = Conv2D(filters=branch3[0], kernel_size=(1,1), padding='same', activation='relu', 
                     kernel_regularizer=l2(0.01))(input)
    conv_23 = Conv2D(filters=branch3[1], kernel_size=(5,5), padding='same', activation='relu', 
                     kernel_regularizer=l2(0.01))(conv_13)

    maxpool_14 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input)
    maxpool_proj_24 = Conv2D(filters=branch4[0], kernel_size=(1,1), strides=(1,1), padding='same', 
                             activation='relu', kernel_regularizer=l2(0.01))(maxpool_14)

    inception_output = concatenate([conv_11, conv_22, conv_23, maxpool_proj_24], axis=3)  # use tf as backend

    return inception_output


if __name__ == "__main__":

    num_classes = 1000
    image_width = 224
    image_height = 224
    channels = 3

    # Assign the values 
    input_shape = (image_width, image_height, channels)

    model = googlenet(input_shape, num_classes)

    model.summary()
