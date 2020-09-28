
# inception_resnet_v2_tf2.py

"""
The inception_resnet a, b and c blocks are 35 x 35, 17 x 17 and 8 x 8 in the gride size. Please note 
that the filters in the joint convoluation for inception_resnet a, b and c blocks are respectively 384, 
1154 and 2048.  

$ python inception_resnet_v2_tf2.py

If users want to run the model, please run the script of Inceptin_v4_func.py. Since it is abstract, 
we do not set the argument of the weights that need to be downloaded from designated weblink. 

Make the the necessary changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 
11.0, cuDNN 8.0.1 and CUDA 450.57. In addition, write the new lines of code to replace the deprecated 
code. 

The script has many changes on the foundation of is Inception v3 by Francios Chollet, Inception v4 by 
Kent Sommers and many other published results. I would like to thank all of them for the contributions. 

Note that the input image format for this model is different than for the VGG16 and ResNet models 
(299x299 instead of 224x224), and that the input preprocessing function is also different (same as 
Xception).

Environment: 
Ubuntu 18.04 
TensorFlow 2.3
Keras 2.4.3
CUDA Toolkit 11.0, 
cuDNN 8.0.1
CUDA 450.57. 

https://arxiv.org/pdf/1602.07261.pdf
"""

import tensorflow as tf 
from tensorflow.keras.layers import Input, Conv2D, Dropout, Dense, Lambda, Flatten, Activation, \
    BatchNormalization, MaxPooling2D, Conv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def inception_stem(input):
    # Define the stem network--keep no chnaged --INception v4
    a = Conv2D(filters=32, kernel_size=(3,3), activation='relu', strides=(2,2))(input)
    a = Conv2D(filters=32, kernel_size=(3,3), activation='relu')(a)
    a = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(a)

    b1 = MaxPooling2D((3,3), strides=(2,2))(a)
    b2 = Conv2D(filters=96, kernel_size=(3,3), activation='relu', strides=(2,2))(a)

    b = concatenate([b1, b2], axis=3)

    c1 = Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='same')(b)
    c1 = Conv2D(filters=96, kernel_size=(3,3), activation='relu')(c1)

    c2 = Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='same')(b)
    c2 = Conv2D(filters=64, kernel_size=(7,1), activation='relu', padding='same')(c2)
    c2 = Conv2D(filters=64, kernel_size=(1,7), activation='relu', padding='same')(c2)
    c2 = Conv2D(filters=96, kernel_size=(3,3), activation='relu', padding='valid')(c2)

    c = concatenate([c1, c2], axis=3)

    d1 = MaxPooling2D((3,3), strides=(2,2))(c)
    d2 = Conv2D(filters=192, kernel_size=(3,3), activation='relu', strides=(2,2))(c)

    d = concatenate([d1, d2], axis=3)
    d = BatchNormalization(axis=3)(d)
    d = Activation('relu')(d)

    return d


def inception_a(input, scale):
    # Define Inception-Resnet-A with Inception v4: 10 iterations 
    e1 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(input)

    e2 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(input)
    e2 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(e2)

    e3 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(input)
    e3 = Conv2D(filters=48, kernel_size=(3,3), activation='relu', padding='same')(e3)
    e3 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(e3)

    e = concatenate([e1, e2, e3], axis=3)

    e = Conv2D(filters=384, kernel_size=(1,1), activation='linear', padding='same')(e)

    if scale: e = Lambda(lambda x: x*0.15)(e)

    e = concatenate([input, e], axis=3)
    e = BatchNormalization(axis=3)(e)
    e = Activation("relu")(e)

    return e


def reduction_a(input):
    # Define Reduction-A: keep no changed--Inception v4
    f1 = MaxPooling2D((3,3), strides=(2,2))(input)

    f2 = Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(2,2))(input)

    f3 = Conv2D(filters=192, kernel_size=(1,1), activation='relu', padding='same')(input)
    f3 = Conv2D(filters=224, kernel_size=(3,3), activation='relu', padding='same')(f3)
    f3 = Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(2,2))(f3)

    f = concatenate([f1, f2, f3], axis=3)
    f = BatchNormalization(axis=3)(f)
    f = Activation('relu')(f)

    return f


def inception_b(input, scale):
    # Define Inception-Resnet-B: 20 iterations 
    g1 = Conv2D(filters=192, kernel_size=(1,1), activation='relu', padding='same')(input)

    g2 = Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same')(input)
    g2 = Conv2D(filters=160, kernel_size=(1,7), activation='relu', padding='same')(g2)
    g2 = Conv2D(filters=192, kernel_size=(7,1), activation='relu', padding='same')(g2)

    g =concatenate([g1, g2], axis=3)

    g = Conv2D(filters=1154, kernel_size=(1,1), activation='linear', padding='same')(g)

    if scale: g = Lambda(lambda x: x*0.10)(g)

    g = concatenate([input, g], axis=3)
    g = BatchNormalization(axis=3)(g)
    g = Activation("relu")(g)

    return g


def reduction_b(input):
    # Adopt the inception_a in the Inception v4
    h1 = MaxPooling2D((3,3), strides=(2,2),padding='valid')(input)

    h2 = Conv2D(filters=256, kernel_size=(1,1), activation='relu', padding='same')(input)
    h2 = Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(2,2))(h2)

    h3 = Conv2D(filters=256, kernel_size=(1,1), activation='relu', padding='same')(input)
    h3 = Conv2D(filters=288, kernel_size=(3,3), activation='relu', strides=(2,2))(h3)

    h4 = Conv2D(filters=256, kernel_size=(1,1), activation='relu', padding='same')(input)
    h4 = Conv2D(filters=288, kernel_size=(3,3), activation='relu', padding='same')(h4)
    h4 = Conv2D(filters=320, kernel_size=(3,3), activation='relu', strides=(2,2))(h4)

    h = concatenate([h1, h2, h3, h4], axis=3)
    h = BatchNormalization(axis=3)(h)
    h = Activation('relu')(h)

    return h


def inception_c(input, scale):
    # Define Inception-Resnet-C: 10 iterations 
    i1 = Conv2D(filters=192, kernel_size=(1,1), activation='relu', padding='same')(input)

    i2 = Conv2D(filters=192, kernel_size=(1,1), activation='relu', padding='same')(input)
    i2 = Conv2D(filters=224, kernel_size=(1,3), activation='relu', padding='same')(i2)
    i2 = Conv2D(filters=256, kernel_size=(3,1), activation='relu', padding='same')(i2)

    i =concatenate([i1, i2], axis=3)

    i = Conv2D(filters=2048, kernel_size=(1,1), activation='linear',padding='same')(i)

    if scale: i = Lambda(lambda x: x*0.20)(i)

    i = concatenate([input, i], axis=3)
    i = BatchNormalization(axis=3)(i)
    i = Activation("relu")(i)

    return i

"""
def aux_output(x):
    # Define an auxiliary classifier 
    y = AveragePooling2D(pool_size=(5,5), strides=(3,3))(x)
    y = Conv2D(filters=128, kernel_size=(1,1),padding='same', activation='relu')(y)
    y = GlobalAveragePooling2D(name='aux_gav_pool')(y)
    y = Dense(units=768, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(num_classes, activation='softmax', name="aux_pred")(y)

    return y 
"""


def inception_resnet_v2(input_shape, num_classes, include_top):
    # Build the inception resnet v2 network
    """
    :param nb_classes: number of classes.txt
    :param scale: flag to add scaling of activations
    :return: 1 input (299x299x3) input shape and 2 outputs (output,aux)
    """
    # Initizate a 3D shape(weight,height,channels) into a 4D tensor(batch, weight, 
    # height, channels). If no batch size, it is defaulted as None.
    inputs = Input(shape=input_shape)

    x = inception_stem(inputs)

    # 10 x Inception Resnet A
    for i in range(10):
        x = inception_a(x, scale=True)

    # Reduction A
    x = reduction_a(x)

    # 20 x Inception Resnet B
    for i in range(20):
        x = inception_b(x, scale=True)

    # Please note it is the auxiliary classifier
    # -y = aux_output(x)

    # Reduction B
    x = reduction_b(x)

    # 10 x Inception Resnet C
    for i in range(10):
        x = inception_c(x, scale=True)

    # Final pooling and prediction
    if include_top:
        x = GlobalAveragePooling2D(name='main_gav_pool')(x)
        x = Dense(units=1536, activation='relu')(x) 
        x = Dropout(0.20)(x)
        x = Dense(units=num_classes, activation='softmax', name='main_pred')(x)

    # Build the model 
    # -model = Model(inputs, [x, y], name='Inception-Resnet-v2')
    model = Model(inputs, x, name='Inception-Resnet-v2')


    return model


if __name__ == '__main__':

    input_shape = (299,299,3)
    num_classes = 1000 
    include_top = True 
    weights = 'imagenet'
    
    model = inception_resnet_v2(input_shape, num_classes, include_top)

    model.summary()