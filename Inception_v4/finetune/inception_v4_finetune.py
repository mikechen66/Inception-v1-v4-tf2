

# inception_v4_finetune.py

"""
The script predicts both the class and the certainty for any designated image. Since the 
prediction-related function is a customized relaization. It is quire different from the 
decode_predictions() within keras. However, the latter only accepts 1000 classes not 1001 
that is defaulted in the Inception V4 Weights. Please give the commands as follows. 

$ python inception_v4_finetune.py

Class is: African elephant, Loxodonta africana
Certainty is: 0.8177135

Uses can change the combination of formal arguments in order to call the back-end model. 
It is useful to fintune the model to realize specific purposes. 

"""

from keras.layers import Input, Conv2D, Dropout, Dense, Flatten, AveragePooling2D
from keras.preprocessing import image
# -from keras.applications.imagenet_utils import decode_predictions
from keras.models import Model
import tensorflow as tf 
import numpy as np
from inception_v4_convbase import inception_stem, inception_a, inception_b, \
    inception_c, reduction_a, reduction_b


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Assume users have already downloaded the Inception v4 weights 
WEIGHTS_PATH = '/home/mike/keras_dnn_models/inception-v4_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = '/home/mike/keras_dnn_models/inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5'


def inception_v4(input_shape, num_classes, weights, include_top):
    # Build the abstract Inception v4 network
    '''
    Args:
        input_shape: three dimensions in the TensorFlow Data Format
        num_classes: number of classes
        weights: pre-defined Inception v4 weights 
        include_top: a boolean, for full traning or finetune 
    Return: 
        logits: the logit outputs of the model.
    '''
    inputs = Input(shape=input_shape)

    # Make the the stem of Inception v4 
    x = inception_stem(inputs)

    # 4 x Inception-A blocks: 35 x 35 x 384
    for i in range(0, 4):
        x = inception_a(x)

    # Reduction-A block: # 35 x 35 x 384
    x = reduction_a(x)

    # 7 x Inception-B blocks: 17 x 17 x 1024
    for i in range(0, 7):
        x = inception_b(x)

    # Reduction-B block: 17 x 17 x 1024
    x = reduction_b(x)

    # 3 x Inception-C blocks: 8 x 8 x 1536
    for i in range(0, 3):
        x = inception_c(x)

    # Final pooling and prediction
    if include_top:
        # 1 x 1 x 1536
        x = AveragePooling2D((8,8), padding='valid')(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(units=num_classes, activation='softmax')(x)

    model = Model(inputs, x, name='inception_v4')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = WEIGHTS_PATH
        else:
            weights_path = WEIGHTS_PATH_NO_TOP
        # -model.load_weights(weights_path, by_name=True)
        model.load_weights(weights_path)

    return model


def preprocess_input(x):
    
    x = image.img_to_array(img)
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
