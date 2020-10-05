
# inception_v1_lrn_aux.py

"""
The script is the commandline-stype sequentual model of GoogLeNet Inception v1 with Local Response 
Normalization and auxilary classifiers. The plain model is used for the thorough study on how Google 
has built the network based on the network depth of AlexNet and the small filter size of NIN(Network 
In Network). It typically stacks the layers for complex expression of the image classification. 

$ inception_v1_lrn_aux.py

With the LRN normalization, the model has the total size of 9+ million. Please see the paper with 
opening the weblink as follows. If removing the LRN, it has the total size of 5+million. 

Make the necessary changes of both the class definition and the context to adapt to the environment 
of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 11.0, cuDNN 8.0.1 and CUDA 450.57. In addition, write 
the new lines of code to replace the deprecated code. 

Thanks for the guys for the contributions. For instance, joelouismarino provided the original script 
of the model in Theano and swghosh modified the script of lrn.py based on the original scrip of 
joelouismarino. Mike modifies it to adatp to TensorFlow 2.2 and Keras 2.4.3

Going Deeper with Convolutions(GoogLeNet Inception v1)
https://arxiv.org/pdf/1409.4842.pdf
"""


import imageio
from PIL import Image
import numpy as np
import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Concatenate, Reshape, Activation

from keras.regularizers import l2
from keras.optimizers import SGD
from lrn import LRN 
from keras import backend


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Define the Googlenet class 
class Googlenet(object):

    # Adopt the static method to enbale the elegant realization of the model  
    @staticmethod
    # Build the GoogLeNet Inveption v1
    def build(input_shape, num_classes):

        input = Input(shape=input_shape)
 
        conv1_7x7 = Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu', name='conv1/7x7_s2')(input)
        pool1_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='pool1/3x3_s2')(conv1_7x7)
        pool1_norm1 = LRN(name='pool1/norm1')( pool1_3x3)
        conv2_3x3_reduce = Conv2D(64, (1,1), padding='valid', activation='relu', name='conv2/3x3_reduce')(pool1_norm1)
        conv2_3x3 = Conv2D(192, (3,3), padding='same', activation='relu', name='conv2/3x3')(conv2_3x3_reduce)
        conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)
        pool2_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='pool2/3x3_s2')(conv2_norm2)

        inception_3a_1x1 = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_3a/1x1')(pool2_3x3)
        inception_3a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='inception_3a/3x3_reduce')(pool2_3x3)
        inception_3a_3x3 = Conv2D(128, (3,3), padding='same', activation='relu', name='inception_3a/3x3')(inception_3a_3x3_reduce)
        inception_3a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='inception_3a/5x5_reduce')(pool2_3x3)
        inception_3a_5x5 = Conv2D(32, (5,5), padding='same', activation='relu', name='inception_3a/5x5')(inception_3a_5x5_reduce)
        inception_3a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_3a/pool')(pool2_3x3)
        inception_3a_pool_proj = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_3a/pool_proj')(inception_3a_pool)
        inception_3a_output = Concatenate(axis=-1, name='inception_3a/output')([inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj])

        inception_3b_1x1 = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_3b/1x1')(inception_3a_output)
        inception_3b_3x3_reduce = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_3b/3x3_reduce')(inception_3a_output)
        inception_3b_3x3 = Conv2D(192, (3,3), padding='same', activation='relu', name='inception_3b/3x3')(inception_3b_3x3_reduce)
        inception_3b_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_3b/5x5_reduce')(inception_3a_output)
        inception_3b_5x5 = Conv2D(96, (5,5), padding='same', activation='relu', name='inception_3b/5x5')(inception_3b_5x5_reduce)
        inception_3b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_3b/pool')(inception_3a_output)
        inception_3b_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_3b/pool_proj')(inception_3b_pool)
        inception_3b_output = Concatenate(axis=-1, name='inception_3b/output')([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj])

        inception_4a_1x1 = Conv2D(192, (1,1), padding='same', activation='relu', name='inception_4a/1x1')(inception_3b_output)
        inception_4a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='inception_4a/3x3_reduce')(inception_3b_output)
        inception_4a_3x3 = Conv2D(208,(3,3), padding='same', activation='relu', name='inception_4a/3x3' ,kernel_regularizer=l2(0.0002))(inception_4a_3x3_reduce)
        inception_4a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='inception_4a/5x5_reduce')(inception_3b_output)
        inception_4a_5x5 = Conv2D(48, (5,5), padding='same', activation='relu', name='inception_4a/5x5')(inception_4a_5x5_reduce)
        inception_4a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4a/pool')(inception_3b_output)
        inception_4a_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4a/pool_proj')(inception_4a_pool)
        inception_4a_output = Concatenate(axis=-1, name='inception_4a/output')([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj])

        loss1_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='loss1/ave_pool')(inception_4a_output)
        loss1_conv = Conv2D(128, (1,1), padding='same', activation='relu', name='loss1/conv')(loss1_ave_pool)
        loss1_fc = Dense(1024, activation='relu', name='loss1/fc')(loss1_conv)
        loss1_drop_fc = Dropout(rate=0.5)(loss1_fc)
        loss1_classifier = Dense(num_classes, name='loss1/classifier')(loss1_drop_fc)
        loss1_classifier_act = Activation('softmax')(loss1_classifier)

        inception_4b_1x1 = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_4b/1x1')(inception_4a_output)
        inception_4b_3x3_reduce = Conv2D(112, (1,1), padding='same', activation='relu', name='inception_4b/3x3_reduce')(inception_4a_output)
        inception_4b_3x3 = Conv2D(224, (3,3), padding='same', activation='relu', name='inception_4b/3x3')(inception_4b_3x3_reduce)
        inception_4b_5x5_reduce = Conv2D(24, (1,1), padding='same', activation='relu', name='inception_4b/5x5_reduce')(inception_4a_output)
        inception_4b_5x5 = Conv2D(64, (5,5), padding='same', activation='relu', name='inception_4b/5x5')(inception_4b_5x5_reduce)
        inception_4b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4b/pool')(inception_4a_output)
        inception_4b_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4b/pool_proj')(inception_4b_pool)
        inception_4b_output = Concatenate(axis=-1, name='inception_4b/output')([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj])
        
        inception_4c_1x1 = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4c/1x1')(inception_4b_output)
        inception_4c_3x3_reduce = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4c/3x3_reduce')(inception_4b_output)
        inception_4c_3x3 = Conv2D(256, (3,3), padding='same', activation='relu', name='inception_4c/3x3')(inception_4c_3x3_reduce)
        inception_4c_5x5_reduce = Conv2D(24, (1,1), padding='same', activation='relu', name='inception_4c/5x5_reduce')(inception_4b_output)
        inception_4c_5x5 = Conv2D(64, (5,5), padding='same', activation='relu', name='inception_4c/5x5')(inception_4c_5x5_reduce)
        inception_4c_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4c/pool')(inception_4b_output)
        inception_4c_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4c/pool_proj')(inception_4c_pool)
        inception_4c_output = Concatenate(axis=-1, name='inception_4c/output')([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj])

        inception_4d_1x1 = Conv2D(112, (1,1), padding='same', activation='relu', name='inception_4d/1x1')(inception_4c_output)
        inception_4d_3x3_reduce = Conv2D(144, (1,1), padding='same', activation='relu', name='inception_4d/3x3_reduce')(inception_4c_output)
        inception_4d_3x3 = Conv2D(288, (3,3), padding='same', activation='relu', name='inception_4d/3x3')(inception_4d_3x3_reduce)
        inception_4d_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_4d/5x5_reduce')(inception_4c_output)
        inception_4d_5x5 = Conv2D(64, (5,5), padding='same', activation='relu', name='inception_4d/5x5')(inception_4d_5x5_reduce)
        inception_4d_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4d/pool')(inception_4c_output)
        inception_4d_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4d/pool_proj')(inception_4d_pool)
        inception_4d_output = Concatenate(axis=-1, name='inception_4d/output')([inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj])
    
        loss2_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='loss2/ave_pool')(inception_4d_output)
        loss2_conv = Conv2D(128, (1,1), padding='same', activation='relu', name='loss2/conv')(loss2_ave_pool)
        loss2_fc = Dense(1024, activation='relu', name='loss2/fc')(loss2_conv)
        loss2_drop_fc = Dropout(rate=0.5)(loss2_fc)
        loss2_classifier = Dense(num_classes, name='loss2/classifier')(loss2_drop_fc)
        loss2_classifier_act = Activation('softmax')(loss2_classifier)

        inception_4e_1x1 = Conv2D(256, (1,1), padding='same', activation='relu', name='inception_4e/1x1')(inception_4d_output)
        inception_4e_3x3_reduce = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_4e/3x3_reduce')(inception_4d_output)
        inception_4e_3x3 = Conv2D(320, (3,3), padding='same', activation='relu', name='inception_4e/3x3')(inception_4e_3x3_reduce)
        inception_4e_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_4e/5x5_reduce')(inception_4d_output)
        inception_4e_5x5 = Conv2D(128, (5,5), padding='same', activation='relu', name='inception_4e/5x5')(inception_4e_5x5_reduce)
        inception_4e_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4e/pool')(inception_4d_output)
        inception_4e_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4e/pool_proj')(inception_4e_pool)
        inception_4e_output = Concatenate(axis=-1, name='inception_4e/output')([inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj])

        inception_5a_1x1 = Conv2D(256, (1,1), padding='same', activation='relu', name='inception_5a/1x1')(inception_4e_output)
        inception_5a_3x3_reduce = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_5a/3x3_reduce')(inception_4e_output)
        inception_5a_3x3 = Conv2D(320, (3,3), padding='same', activation='relu', name='inception_5a/3x3')(inception_5a_3x3_reduce)
        inception_5a_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_5a/5x5_reduce')(inception_4e_output)
        inception_5a_5x5 = Conv2D(128, (5,5), padding='same', activation='relu', name='inception_5a/5x5')(inception_5a_5x5_reduce)
        inception_5a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_5a/pool')(inception_4e_output)
        inception_5a_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_5a/pool_proj')(inception_5a_pool)
        inception_5a_output = Concatenate(axis=-1, name='inception_5a/output')([inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj])

        inception_5b_1x1 = Conv2D(384, (1,1), padding='same', activation='relu', name='inception_5b/1x1')(inception_5a_output)
        inception_5b_3x3_reduce = Conv2D(192, (1,1), padding='same', activation='relu', name='inception_5b/3x3_reduce')(inception_5a_output)
        inception_5b_3x3 = Conv2D(384, (3,3), padding='same', activation='relu', name='inception_5b/3x3')(inception_5b_3x3_reduce)
        inception_5b_5x5_reduce = Conv2D(48, (1,1), padding='same', activation='relu', name='inception_5b/5x5_reduce')(inception_5a_output)
        inception_5b_5x5 = Conv2D(128, (5,5), padding='same', activation='relu', name='inception_5b/5x5')(inception_5b_5x5_reduce)
        inception_5b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_5b/pool')(inception_5a_output)
        inception_5b_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_5b/pool_proj')(inception_5b_pool)
        inception_5b_output = Concatenate(axis=-1, name='inception_5b/output')([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj])

        pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7), strides=(1,1), name='pool5/7x7_s2')(inception_5b_output)
        pool5_drop_7x7_s1 = Dropout(rate=0.5)(pool5_7x7_s1)
        loss3_classifier = Dense(num_classes, name='loss3/classifier')(pool5_drop_7x7_s1)
        loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

        inception_v1 = Model(inputs=input, outputs=[loss1_classifier_act, loss2_classifier_act, loss3_classifier_act])

        return inception_v1


if __name__ == "__main__":

    input_shape = (224, 224, 3)
    num_classes = 1000

    inception_v1 = Googlenet.build(input_shape, num_classes)

    inception_v1.summary()
