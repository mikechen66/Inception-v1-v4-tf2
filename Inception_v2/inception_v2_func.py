#!/usr/bin/env python
# coding: utf-8

# incepion_v2_func.py

"""
The script is the function-style model of GoogLeNet Inception v2. The second-stage model is used 
for the thorough study on how Google has built the network based with BatchNorm on the network 
depth of AlexNet and the small filter size of NIN(Network In Network). It stacks the layers to 
address the complexity of the image classification. Please use the following command to run the 
script. 

$ python inception_v2_func.py

It is quite strange it is hard to get the code of Inception v2 in either Google Search or Github. 
Futhermotre, most of the available Inception v2 has more than total size of 20+ million (or even 
200 mllion) parameters in the condition of 1000 classes and input size of (224,224,3). That is 
amazing. In contrast, the official Inception v2 has lass than 10 million. Therefore, the new-built
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

    conv1_7x7 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu', 
                       kernel_regularizer=l2(0.01))(input)
    bn_1 = BatchNormalization()(conv1_7x7)
    maxpool1_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(bn_1)
    
    conv2_1x1 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', 
                              kernel_regularizer=l2(0.01))(maxpool1_3x3)
    bn_2 = BatchNormalization()(conv2_1x1)
    
    conv2_3x3 = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', 
                       kernel_regularizer=l2(0.01))(bn_2)
    bn3 = BatchNormalization()(conv2_3x3)
    maxpool2_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(bn3)

    inception_3a = inception(input=maxpool2_3x3, axis=3, params=[(64,),(64,64),(64,96,96),(32,)])
    inception_3b = inception(input=inception_3a, axis=3, params=[(64,),(64,64),(64,96,96),(64,)])
    maxpool3_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(inception_3b)

    inception_4a = inception(input=maxpool3_3x3, axis=3, params=[(224,),(64,96),(96,128,128),(128,)])
    inception_4b = inception(input=inception_4a, axis=3, params=[(192,),(96,128),(96,128,128),(128,)])
    inception_4c = inception(input=inception_4b, axis=3, params=[(160,),(128,160),(128,160,160),(96,)])
    inception_4d = inception(input=inception_4c, axis=3, params=[(96,),(128,192),(160,192,192),(96,)])
    maxpool4_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(inception_4d)

    inception_5a = inception(input=maxpool4_3x3, axis=3, params=[(352,),(192,320),(160,224,224),(128,)])
    inception_5b = inception(input=inception_5a, axis=3, params=[(352,),(192,320),(192,224,224),(128,)]) 
    avgpool1_7x7 = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(inception_5b)

    drop = Dropout(rate=0.4)(avgpool1_7x7)
    linear = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(drop)

    model = Model(inputs=input, outputs=linear)

    return model 


def inception(input, axis, params):

    # Bind the horizontal cells tegother for an elegant realization 
    [branch1, branch2, branch3, branch4] = params

    # Layer1-->No.1: Conv 64 1x1 + 1(s)
    conv_11 = Conv2D(filters=branch1[0], kernel_size=(1,1), padding='same', activation='relu', 
                          kernel_regularizer=l2(0.01))(input)
    bn_11 = BatchNormalization()(conv_11)

    # Layer1-->No.2: Conv 64 1x1 +１(s), be connected to conv_3x3_l21 (Layer2-->No.1)
    conv_12 = Conv2D(filters=branch2[0], kernel_size=(1,1), padding='same', activation='relu', 
                                 kernel_regularizer=l2(0.01))(input)
    bn_12 = BatchNormalization()(conv_12)
    # Layer2-->No.2: Conv 64 3x3 + 1(s), conv_1x1_reduce is the cell of Inception Layer 2
    conv_22 = Conv2D(filters=branch2[1], kernel_size=(3,3), padding='same', activation='relu', 
                     kernel_regularizer=l2(0.01))(bn_12)
    bn_22 = BatchNormalization()(conv_22)

    # Layer1-->No.3: Conv 64 1x1 1(s), conv_3x3_reduce_l13 is the cell of Inception Layer 1
    conv_13 = Conv2D(filters=branch3[0], kernel_size=(1,1), padding='same', activation='relu', 
                     kernel_regularizer=l2(0.01))(input)
    bn_13 = BatchNormalization()(conv_13)
    # Layer2-->No.3: Conv 96 3x3 1(s)
    conv_23 = Conv2D(filters=branch3[1], kernel_size=(3,3), padding='same', activation='relu', 
                     kernel_regularizer=l2(0.01))(bn_13)
    bn_23 = BatchNormalization()(conv_23)
    # Layer3-->No.3: Conv 96 3x3 1(s): The Layer 3 has only one cell 
    conv_33 = Conv2D(filters=branch3[2], kernel_size=(3,3), padding='same', activation='relu', 
                     kernel_regularizer=l2(0.01))(bn_23)
    bn_33 = BatchNormalization()(conv_33)

    # Layer1-->No.4: AvgPool 3x3 1(s)
    avgpool_14 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input)
    # Layer2-->No.4
    avgpool_proj_24 = Conv2D(filters=branch4[0], kernel_size=(1,1), strides=(1,1), padding='same', 
                             activation='relu', kernel_regularizer=l2(0.01))(avgpool_14)
    bn_24 = BatchNormalization()(avgpool_proj_24)
    
    inception_output = concatenate([bn_11, bn_22, bn_33, bn_24], axis=3)

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
