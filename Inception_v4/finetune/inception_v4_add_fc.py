#!/usr/bin/env python
# coding: utf-8

# inception_v4_add_fc.py

"""
The script predicts both the class and the certainty for any designated image. Since the 
prediction-related function is a customized relaization. It is quire different from the 
decode_predictions() within keras. However, the latter only accepts 1000 classes not 1001 
that is defaulted in the Inception V4 Weights. Please give the commands as follows. 

$ python inception_v4_add_fc.py

Class is: African elephant, Loxodonta africana
Certainty is: 0.8177135

Uses can change the combination of formal arguments in order to call the back-end model. 
It is useful to fintune the model to realize specific purposes. 
"""


from keras.layers import Dense, Flatten, AveragePooling2D
from keras.preprocessing import image
# -from keras.applications.imagenet_utils import decode_predictions
from keras import models 
from keras import layers 
import tensorflow as tf 
import numpy as np
from inception_v4_conv_fc import inception_v4


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


input_shape = (299,299,3)
num_classes = 1001
weights='imagenet'
include_top = None 

conv_base = inception_v4(input_shape, num_classes, weights, include_top)

model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D(name='avg_pool'))
model.add(layers.Dropout(0.6))
model.add(layers.Flatten())
model.add(layers.Dense(units=1000, activation='softmax'))
model.summary()