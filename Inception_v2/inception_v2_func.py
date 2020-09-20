#!/usr/bin/env python
# coding: utf-8

# incepion_v1_func.py

"""
The script is the function-style model of GoogLeNet Inception v2. The second-stage model is used 
for the thorough study on how Google has built the network based with BatchNorm on the network 
depth of AlexNet and the small filter size of NIN(Network In Network). It stacks the layers to 
address the complexity of the image classification. Please use the following command to run the 
script. It is much easier to understand than the slim model of Inception v2 in the official 
TensorFlow models. The slim model includes too much syntactic sugar. 

# $ python inception_v2_func.py

It is quite strange it is hard to get the code of Inception v2 in either Google Search or Github. 
Futhermotre, most of the available Inception v2 has more than total size of 20+ million (or even 
200 mllion) parameters in the condition of 1000 classes and input size of (224,224,3). That is 
amazing. In contrast, the official Inception v2 has lass than 10+ million. Therefore, the new-built
script is deidicated to provide users with the accurate model incarnation with the total size of 
parameters to 8+ million. It is quite close to the official published parameter size by GoogleNet 
Inception Team. 

Notes:

Please remmeber that it is the TensorFlow realization with image_data_foramt = 'channels_last'. If
the env of Keras is 'channels_first', please change to the TensorFlow convention. 

The realizatin is eight ineption layers after deducting the inception_4e. In contrast, the script 
of Inception v1 includes the inception_4e. If users want more layers, it is necessary to comply 
with the prerequize of the following paper of Inception. 

Environment: 

Ubuntu 18.04 
TensorFlow 2.3
Keras 2.4.3
CUDA Toolkit 11.0, 
cuDNN 8.0.1
CUDA 450.57. 

Reference: 
Rethinking the Inception Architecture for Computer Vision(GoogLeNet Inception v2)
https://arxiv.org/pdf/1512.00567v3.pdf
"""


from keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.models import Model
from keras.layers.merge import concatenate
from keras.regularizers import l2


def googlenet(input_shape, num_classes):

    input = Input(shape=input_shape)

    conv1_7x7_s2 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    maxpool1_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv1_7x7_s2)
    BatchNorm_g1 = BatchNormalization()(maxpool1_3x3_s2)
    
    conv2_3x3_reduce = Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(BatchNorm_g1)
    BatchNorm_g2 = BatchNormalization()(conv2_3x3_reduce)
    
    conv2_3x3 = Conv2D(filters=192, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(BatchNorm_g2)
    BatchNorm_g3 = BatchNormalization()(conv2_3x3)
    maxpool2_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(BatchNorm_g3)

    inception_3a = inception(input=maxpool2_3x3_s2, filters_1x1_l11=64, filters_3x3_reduce_l12=64, filters_3x3_l21=64, filters_3x3_reduce_l13=64, filters_3x3_l22=96, filters_3x3_l31=96, filters_pool_proj=32)
    inception_3b = inception(input=inception_3a, filters_1x1_l11=64, filters_3x3_reduce_l12=64, filters_3x3_l21=64, filters_3x3_reduce_l13=64, filters_3x3_l22=96, filters_3x3_l31=96, filters_pool_proj=64)
    maxpool3_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(inception_3b)

    inception_4a = inception(input=maxpool3_3x3_s2, filters_1x1_l11=224, filters_3x3_reduce_l12=64, filters_3x3_l21=96, filters_3x3_reduce_l13=96, filters_3x3_l22=128, filters_3x3_l31=128, filters_pool_proj=128)
    inception_4b = inception(input=inception_4a, filters_1x1_l11=192, filters_3x3_reduce_l12=96, filters_3x3_l21=128, filters_3x3_reduce_l13=96, filters_3x3_l22=128, filters_3x3_l31=128, filters_pool_proj=128)
    inception_4c = inception(input=inception_4b, filters_1x1_l11=160, filters_3x3_reduce_l12=128, filters_3x3_l21=160, filters_3x3_reduce_l13=128, filters_3x3_l22=160, filters_3x3_l31=160, filters_pool_proj=96)
    inception_4d = inception(input=inception_4c, filters_1x1_l11=96, filters_3x3_reduce_l12=128, filters_3x3_l21=192, filters_3x3_reduce_l13=160, filters_3x3_l22=192, filters_3x3_l31=192, filters_pool_proj=96)
    # -inception_4e = inception(input=inception_4d, filters_1x1_l11=64, filters_3x3_reduce_l12=64, filters_3x3_l21=64, filters_3x3_reduce_l13=64, filters_3x3_l22=96, filters_3x3_l31=96, filters_pool_proj=64)
    maxpool4_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(inception_4d)

    inception_5a = inception(input=maxpool4_3x3_s2, filters_1x1_l11=352, filters_3x3_reduce_l12=192, filters_3x3_l21=320, filters_3x3_reduce_l13=160, filters_3x3_l22=224, filters_3x3_l31=224, filters_pool_proj=128)
    inception_5b = inception(input=inception_5a, filters_1x1_l11=352, filters_3x3_reduce_l12=192, filters_3x3_l21=320, filters_3x3_reduce_l13=192, filters_3x3_l22=224, filters_3x3_l31=224, filters_pool_proj=128)
    
    averagepool1_7x7_s1 = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(inception_5b)
    drop = Dropout(rate=0.4)(averagepool1_7x7_s1)
    linear = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(drop)

    model = Model(inputs=input, outputs=linear)

    return model 


def inception(input, filters_1x1_l11, filters_3x3_reduce_l12, filters_3x3_l21, filters_3x3_reduce_l13, filters_3x3_l22, filters_3x3_l31, filters_pool_proj):

    # Layer1-->No.1: Conv 64 1x1 + 1(s)
    conv_1x1_l11 = Conv2D(filters=filters_1x1_l11, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    BatchNorm_i1 = BatchNormalization()(conv_1x1_l11)
    # Layer1-->No.2: Conv 64 1x1 +１(s), be connected to conv_3x3_l21 (Layer2-->No.1)
    conv_3x3_reduce_l12 = Conv2D(filters=filters_3x3_reduce_l12, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    BatchNorm_i2 = BatchNormalization()(conv_3x3_reduce_l12)
    # Layer2-->No.1: Conv 64 3x3 + 1(s), conv_3x3_reduce is the cell of Inception Layer 2
    conv_3x3_l21 = Conv2D(filters=filters_3x3_l21, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(BatchNorm_i2)
    BatchNorm_i3 = BatchNormalization()(conv_3x3_l21)
    # Layer1-->No.3: Conv 64 1x1 1(s), conv_3x3_reduce_l13 is the cell of Inception Layer 1
    conv_3x3_reduce_l13 = Conv2D(filters=filters_3x3_reduce_l13, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    BatchNorm_i4 = BatchNormalization()(conv_3x3_reduce_l13)
    # Layer2-->No.2: Conv 96 3x3 1(s)
    conv_3x3_l22 = Conv2D(filters=filters_3x3_l22, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(BatchNorm_i4)
    BatchNorm_i5 = BatchNormalization()(conv_3x3_l22)
    # Layer3-->No.1: Conv 96 3x3 1(s): The Layer 3 has only one cell 
    conv_3x3_l31 = Conv2D(filters=filters_3x3_l31, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(BatchNorm_i5)
    BatchNorm_i6 = BatchNormalization()(conv_3x3_l31)
    # Layer1-->No.4: AvgPool 3x3 1(s)
    avgpool_14 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input)
    # Layer2-->No.3
    avgpool_23_proj = Conv2D(filters=filters_pool_proj, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(avgpool_14)
    BatchNorm_i7 = BatchNormalization()(avgpool_23_proj)
    # -inception_output = concatenate([conv_1x1_l11, conv_3x3_l21, conv_3x3_l31, avgpool_23_proj], axis=3) 
    inception_output = concatenate([BatchNorm_i1, BatchNorm_i3, BatchNorm_i6, BatchNorm_i7], axis=3)

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