# -*- coding: utf-8 -*-
"""

Inception V3 model for Keras. Please make the following command in the Linux Terminal and get 
the related predicted resulr. 

$ python inception_v3_predictg.py

Predicted: [[('n02504458', 'African_elephant', 0.9749893), ('n01871265', 'tusker', 0.016361168),
('n02504013', 'Indian_elephant', 0.002354787), ('n03633091', 'ladle', 0.000102942366), ('n04596742', 
'wok', 9.29997e-05)]]

Editor: Mike Chen

Make the the necessary changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 
11.0, cuDNN 8.0.1 and CUDA 450.57. In addition, write the new lines of code to replace the deprecated code. 


Author: Francios Chollet -- inception_v3.py 
Note that the input image format for this model is different than for the VGG16 and ResNet models 
(299x299 instead of 224x224), and that the input preprocessing function is also different (same as 
Xception).

Reference:
[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
"""

import warnings
import numpy as np
import tensorflow as tf 
from keras.preprocessing import image
from keras.models import Model

from keras import backend as K
from keras import layers
from keras.layers import Input, Conv2D, SeparableConv2D, Dense, Activation, BatchNormalization, \
    MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from imagenet_utils import _obtain_input_shape


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Give the paths for the two types of weights  
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

def stem(input):
    # Define the stem network 
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    return x 


def conv_a(input): 
    # Definr the function of conv_a that has three iterations 

    # Interation 1: 
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(input)
    residual = BatchNormalization()(residual)
    
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(input)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    # Iteration 2: 
    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    # Iteration 3: 
    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    return x 


def conv_b(input, i): 

    residual = input
    prefix = 'block' + str(i + 5)

    x = Activation('relu', name=prefix + '_sepconv1_act')(input)

    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
    x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
    x = Activation('relu', name=prefix + '_sepconv2_act')(x)

    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
    x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
    x = Activation('relu', name=prefix + '_sepconv3_act')(x)

    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
    x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

    x = layers.add([x, residual])
    
    return x 


def conv_c(input):

    residual = Conv2D(1024, (1,1), strides=(2,2), padding='same', use_bias=False)(input)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(input)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    return x 


def xception(include_top=True, weights='imagenet', input_tensor=None,
             input_shape=None, pooling=None, num_classes=1000):
    # Instantiates the Xception architecture.
    """
    # Arguments
        include_top: whether to include the FC layer at the top of the network.
        weights: `None` (random initialization) or "imagenet" 
        input_tensor: Keras tensor, i.e., output of `layers.Input()`
        input_shape: tuple, specified if `include_top` is False or `(299,299,3)`.
        pooling: Optional mode for feature extraction when `include_top` is `False`.
            - `None` means the output of the model as the 4D tensor.
            - `avg` means lobal average pooling applied to the output of the last 
              conv layer and the output as a 2D tensor.
            - `max` means global max pooling to be applied.
        num_classes: optional number of classes to classify images. 
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`, or invalid input shape.
        RuntimeError: If attempt to run this model with a backend, it does not support
        separable convolutions.
    """

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape, default_size=299, min_size=139, 
                                      data_format=None, weights=weights,
                                      require_flatten=include_top)
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Ensure that the model considers any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Call the stem function
    x = stem(inputs)

    # Already have the 3 iterartions 
    x = conv_a(x)

    # 8 x conv_b
    for i in range(0, 8): 
        x = conv_b(x,i)

    # Call the function of conv_c
    x = conv_c(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(num_classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Build the model
    model = Model(inputs, x, name='xception')

    # Load the weights 
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
    
    return model


def preprocess_input(x):
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    output = np.multiply(x, 2.0)

    return output 


if __name__ == '__main__':

    input_shape = (229, 299, 3)
    model = xception(input_shape)
    model.summary()

    img_path = '/home/mike/Documents/keras_xception_v1/elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    output = preprocess_input(img)
    print('Input image shape:', output.shape)

    preds = model.predict(output)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds, 1))