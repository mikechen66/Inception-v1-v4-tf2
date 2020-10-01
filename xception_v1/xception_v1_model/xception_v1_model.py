 #!/usr/bin/env python
# coding: utf-8

# xception_v1_model.py

"""
Xception v1  model for Keras. 

Rebuild the Xception v1 model with the function-oriented programming. It addresses the power, elagance 
and simplicity with the structured functions. Please make the following command in the Linux Terminal 
and get the related predicted result. 

$ python model_predict.py

Please note that the first iteration of conv_a does not include the typical head of activation. So 
the fucntion of range() is not added into the section. In contrast, the section of conv_b is well 
structured to address the eight itertations. 

This model is available for TensorFlow only, and can only be used with inputs following the TensorFlow 
data format `(width, height, channels)`. You have to set `image_data_format="channels_last"` in your 
Keras config located at ~/.keras/keras.json. 

On ImageNet, this model gets to a top-1 validation accuracy of 0.790 and a top-5 validation accuracy of 
0.945. Also do note that this model is only available for the TensorFlow backend, due to its reliance on 
`SeparableConvolution` layers.

Make the the necessary changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 
11.0, cuDNN 8.0.1 and CUDA 450.57. In addition, write the new lines of code to replace the deprecated 
code.

Environment: 

Ubuntu 18.04 
TensorFlow 2.3
Keras 2.4.3
CUDA Toolkit 11.0, 
cuDNN 8.0.1
CUDA 450.57.
Reference:
[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
"""

import tensorflow as tf 
from keras.models import Model
from keras import backend as K

from keras import layers
from keras.layers import Input, Conv2D, SeparableConv2D, Dense, Activation, BatchNormalization, \
    MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from imagenet_utils import _obtain_input_shape


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Give the paths for two types of weights  
WEIGHTS_PATH = '/home/mike/keras_dnn_models/xception_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = '/home/mike/keras_dnn_models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


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
    
    # Exclude the head of activation of 'relu' because it is connected to the above activation. 
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

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = WEIGHTS_PATH
        else:
            weights_path = WEIGHTS_PATH_NO_TOP
        # -model.load_weights(weights_path, by_name=True)
        model.load_weights(weights_path)
    
    return model
