#!/usr/bin/env python
# coding: utf-8

# pred.py

"""
Xception V1 model for Keras.

This model is available for TensorFlow only, and can only be used with inputs following the TensorFlow 
data format `(width, height, channels)`. You have to set `image_data_format="channels_last"` in your 
Keras config located at ~/.keras/keras.json. 

On ImageNet, this model gets to a top-1 validation accuracy of 0.790 and a top-5 validation accuracy of 
0.945. Also do note that this model is only available for the TensorFlow backend, due to its reliance on 
`SeparableConvolution` layers. Please run the command as follows. 

$ predict.py

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


import numpy as np
import tensorflow as tf 
from keras.preprocessing import image
from xception_v1_func import xception 
from keras.applications.imagenet_utils import decode_predictions


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def preprocess_input(x):
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    output = np.multiply(x, 2.0)

    return output 


if __name__ == '__main__':

    input_shape = (229,299,3)
    
    model = xception(input_shape)

    model.summary()

    img_path = '/home/mike/Documents/keras_xception_v1/elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    output = preprocess_input(img)
    print('Input image shape:', output.shape)

    preds = model.predict(output)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds,1))
