

# predict.py

"""
Predict five classses for any designated image 
"""

import tensorflow as tf 
import numpy as np
from inception_v3 import Inception_V3
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


img_path = '/home/mike/Documents/keras_inception_v3/elephant.jpg'
img = image.load_img(img_path, target_size=(299,299))
x = image.img_to_array(img)


def preprocess_input(x):
 
    x = np.expand_dims(x, axis=0)
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)

    return x


if __name__ == '__main__':

    x = preprocess_input(x)
    model = Inception_V3(include_top=True, weights='imagenet')
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))