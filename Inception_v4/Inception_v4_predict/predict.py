

# predict.py

"""
The script predicts both the class and the certainty for any designated image. Since the 
prediction-related function is a customized relaization. It is quire different from the 
decode_predictions() within keras. However, the latter only accepts 1000 classes not 1001 t
hat is defaulted in the Inception V4 Weights. Please give the commands as follows. 

$ python predict.py

Class is: African elephant, Loxodonta africana
Certainty is: 0.8177135
"""

import tensorflow as tf 
import numpy as np
from keras.preprocessing import image
# -from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from inception_v4_func import inception_v4

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

    input_shape = (299,299,3)
    num_classes = 1001
    weights='imagenet'
    include_top = True 

    model = inception_v4(input_shape, num_classes, weights, include_top)

    model.summary()

    img_path = '/home/mike/Documents/keras_inception_v4/elephant.jpg'
    img = image.load_img(img_path, target_size=(299,299))
    output = preprocess_input(img)

    # Open the class label dictionary(that is a human readable label-given ID)
    classes = eval(open('/home/mike/Documents/keras_inception_v4/validation_utils/class_names.txt', 'r').read())

    # Run the prediction on the given image
    preds = model.predict(output)
    print("Class is: " + classes[np.argmax(preds)-1])
    print("Certainty is: " + str(preds[0][np.argmax(preds)]))
