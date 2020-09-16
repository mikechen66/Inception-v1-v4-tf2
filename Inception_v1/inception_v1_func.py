

# incepion_v1func.py

"""
The script is the function-style sequentual model of GoogLeNet Inception v1. The first-stage model 
is used for the thorough study on how Google has built the network based on the network depth of 
AlexNet and the small filter size of NIN(Network In Network). It stacks the layers to address the 
complexity of the image classification. Please use the following command to run the script. 

# $ python inception_v1_func.py

It is quite strange that most of the available Inception v1 has more than total size of 10+ million 
parameters. In contrast, the official Inception v1 has only the 5+ million. Therefore, the modified 
script downsizes the total size of parameters to 6+ million. It is close to the official published
parameter size. 

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


from keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras.layers.merge import concatenate
from keras.regularizers import l2



def googlenet(input_shape, num_classes):

    input = Input(shape=input_shape)

    conv1_7x7_s2 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    maxpool1_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv1_7x7_s2)
    conv2_3x3_reduce = Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(maxpool1_3x3_s2)
    conv2_3x3 = Conv2D(filters=192, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv2_3x3_reduce)
    maxpool2_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv2_3x3)

    inception_3a = inception(input=maxpool2_3x3_s2, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32)
    inception_3b = inception(input=inception_3a, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64)
    maxpool3_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(inception_3b)

    inception_4a = inception(input=maxpool3_3x3_s2, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64)
    inception_4b = inception(input=inception_4a, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)
    inception_4c = inception(input=inception_4b, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256, filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)
    inception_4d = inception(input=inception_4c, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288, filters_5x5_reduce=32, filters_5x5=64, filters_pool_proj=64)
    inception_4e = inception(input=inception_4d, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128)
    maxpool4_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(inception_4e)

    inception_5a = inception(input=maxpool4_3x3_s2, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128)
    inception_5b = inception(input=inception_5a, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384, filters_5x5_reduce=48, filters_5x5=128, filters_pool_proj=128)
    averagepool1_7x7_s1 = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(inception_5b)
    drop = Dropout(rate=0.4)(averagepool1_7x7_s1)

    linear = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(drop)
    
    model = Model(inputs=input, outputs=linear)

    return model 


def inception(input, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):

    conv_1x1 = Conv2D(filters=filters_1x1, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_3x3_reduce = Conv2D(filters=filters_3x3_reduce, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_3x3 = Conv2D(filters=filters_3x3, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv_3x3_reduce)
    conv_5x5_reduce = Conv2D(filters=filters_5x5_reduce, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_5x5 = Conv2D(filters=filters_5x5, kernel_size=(5,5), padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv_5x5_reduce)
    maxpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input)
    maxpool_proj = Conv2D(filters=filters_pool_proj, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(maxpool)
    inception_output = concatenate([conv_1x1, conv_3x3, conv_5x5, maxpool_proj], axis=3)  # use tf as backend

    return inception_output


if __name__ == "__main__":

    num_classes = 1000
    image_width = 224
    image_height = 224
    channels = 3

    # Assign the values 
    input_shape = (image_width, image_height, channels)

    model = googlenet(input_shape, num_classes)

    model.summary()