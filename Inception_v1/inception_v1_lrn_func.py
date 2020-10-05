#!/usr/bin/env python
# coding: utf-8

# incepion_v1_lrn_func.py

"""
The script is the function-style sequentual model of GoogLeNet Inception v1. The first-stage model 
is used for the thorough study on how Google has built the network based on the network depth of 
AlexNet and the small filter size of NIN(Network In Network). It stacks the layers to address the 
complexity of the image classification. Please use the following command to run the script. 

$ python inception_v1_lrn_func.py

If adding the auxilary classifiers, the Inception v1 has the total size of 9+ million parameters. 
In contrast, the Inception v1 without auxiliary classifiers has only 6+ million. It is close to 
the official total size of parameters released by the GoogleNet team. 

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

import tensorflow as tf 
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Lambda, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Define the LRN inhereted from Lambda  
class LRN(Lambda):

    def __init__(self, alpha=0.0001, beta=0.75, depth_radius=5, **kwargs):
        # using parameter defaults as per GoogLeNet
        params = {
            "alpha": alpha,
            "beta": beta,
            "depth_radius": depth_radius
        }
        # Construct a function for use with Keras Lambda
        lrn_fn = lambda inputs: tf.nn.local_response_normalization(inputs, **params)

        # Pass the function to Keras Lambda
        return super().__init__(lrn_fn, **kwargs)


def googlenet(input_shape, num_classes):

    input = Input(shape=input_shape)

    conv1_7x7 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu', 
                       kernel_regularizer=l2(0.01))(input)
    maxpool1_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv1_7x7)
    pool1_norm1 = LRN()(maxpool1_3x3)
    conv2_1x1 = Conv2D(filters=64, kernel_size=(1,1),  strides=(1,1), padding='valid', activation='relu', 
                       kernel_regularizer=l2(0.01))(pool1_norm1)
    conv2_3x3 = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', 
                       kernel_regularizer=l2(0.01))(conv2_1x1)
    conv2_norm2 = LRN()(conv2_3x3)
    maxpool2_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv2_norm2)

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

    inception_output = concatenate([conv_11, conv_22, conv_23, maxpool_proj_24], axis=3)  

    return inception_output


if __name__ == "__main__":

    input_shape = (224, 224, 3)
    num_classes = 1000

    # Assign the values 
    model = googlenet(input_shape, num_classes)

    model.summary()
