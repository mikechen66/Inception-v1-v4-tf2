#!/usr/bin/env python
# coding: utf-8

# incepion_v1_func.py

"""
The script is the function-style sequentual model of GoogLeNet Inception v1. The first-stage model is used for the thorough study on how 
Google has built the network based on the network depth of AlexNet and the small filter size of NIN(Network In Network). It stacks the 
layers to address the complexity of the image classification. Please use the following command to run the script. 

$ python inception_v1_func.py

It is quite strange that most of the available Inception v1 has more than total size of 10+ million parameters. In contrast, the official 
Inception v1 has only the 5+ million. Therefore, the modified script downsizes the total size of parameters to 6+ million. It is close to 
the official published parameter size. 

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

    conv1_7x7 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    maxpool1_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv1_7x7)
    conv2_1x1 = Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(maxpool1_3x3)
    conv2_3x3 = Conv2D(filters=192, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv2_1x1)
    maxpool2_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv2_3x3)

    inception_3a = inception(input=maxpool2_3x3, filters_11=64, filters_12=96, filters_22=128, filters_13=16, filters_23=32, filters_24=32)
    inception_3b = inception(input=inception_3a, filters_11=128, filters_12=128, filters_22=192, filters_13=32, filters_23=96, filters_24=64)
    maxpool3_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(inception_3b)

    inception_4a = inception(input=maxpool3_3x3, filters_11=192, filters_12=96, filters_22=208, filters_13=16, filters_23=48, filters_24=64)
    inception_4b = inception(input=inception_4a, filters_11=160, filters_12=112, filters_22=224, filters_13=24, filters_23=64, filters_24=64)
    inception_4c = inception(input=inception_4b, filters_11=128, filters_12=128, filters_22=256, filters_13=24, filters_23=64, filters_24=64)
    inception_4d = inception(input=inception_4c, filters_11=112, filters_12=144, filters_22=288, filters_13=32, filters_23=64, filters_24=64)
    inception_4e = inception(input=inception_4d, filters_11=256, filters_12=160, filters_22=320, filters_13=32, filters_23=128, filters_24=128)
    maxpool4_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(inception_4e)

    inception_5a = inception(input=maxpool4_3x3, filters_11=256, filters_12=160, filters_22=320, filters_13=32, filters_23=128, filters_24=128)
    inception_5b = inception(input=inception_5a, filters_11=384, filters_12=192, filters_22=384, filters_13=48, filters_23=128, filters_24=128)
    avgpool1_7x7 = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(inception_5b)
    
    drop = Dropout(rate=0.4)(avgpool1_7x7)
    linear = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(drop)
    
    model = Model(inputs=input, outputs=linear)

    return model 


def inception(input, filters_11, filters_12, filters_22, filters_13, filters_23, filters_24):

    conv_11 = Conv2D(filters=filters_11, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_12 = Conv2D(filters=filters_12, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_22 = Conv2D(filters=filters_22, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv_12)
    conv_13 = Conv2D(filters=filters_13, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_23 = Conv2D(filters=filters_23, kernel_size=(5,5), padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv_13)
    maxpool_14 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input)
    maxpool_proj_24 = Conv2D(filters=filters_24, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', 
                             kernel_regularizer=l2(0.01))(maxpool_14)
    
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
