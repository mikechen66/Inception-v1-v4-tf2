#!/usr/bin/env python
# coding: utf-8

# inception_v4_finetune.py

"""
Finetuning consists of unfreezing a few of the top layers of a frozen model base used for 
feature extraction, and joint training both the newly added part of the model, the fully 
connected classifier and the top layers. This is called finetuning because it slightly 
adjusts the more abstract representations of the model being reused in order to make them 
more relevant for the problem at hand.

$ python inception_v4_finetune.py

We set the GPU as 4096 MiB to avoid the runtime error on RTX 2070 Super 2070. If users has 
a big GPU, users can optionally choose the following setting. 

# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
tf.config.experimental.set_memory_growth(gpu, True)

It is worth noting the finetune could not achive higher accruacy rbecuase its adoption a small 
databset of 4000 images. If users adopt the larger dataset such as the original cats_vs_dogs, 
it woule have a higher accuracy. Since we use binary_crossentropy loss, we need binary labels. 
It is worth noting that the argument of lr=le-4 could not improve the accuracy in the function 
of model.compile().  

optimizer=optimizers.RMSprop(lr=1e-5)
"""


import os 
import datetime
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

from keras import models, layers, optimizers
from keras.layers import Dense, Flatten, AveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from inception_v4_conv_fc import inception_v4
from numba import cuda


# Set up the GPU memory size to avoid the out-of-memory error
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


# Give the directory for train, validation and test sets. 
base_dir = '/home/mike/Documents/keras_inception_v4/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


# Give the arguments for the conv_base
input_shape = (299,299,3)
num_classes = 1001
weights='imagenet'
include_top = None 

conv_base = inception_v4(input_shape, num_classes, weights, include_top)
conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D(name='avg_pool'))
model.add(layers.Dropout(0.6))
model.add(layers.Flatten())
model.add(layers.Dense(units=2, activation='softmax'))
model.summary()


print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))


# Free the conv_base before the training 
conv_base.trainable = False


print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))


# Train the given datasets with data generator
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(299,299),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(299,299),
        batch_size=20,
        class_mode='binary')


# Finetune the last fully connected layers(the inception_c block)
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'inception_c':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


# Compile the model 
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])


# Start Tensorboard --logdir logs/fit
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
callback_list = [tensorboard_callback]

# Train the fintuned model 
history = model.fit(train_generator,
                    steps_per_epoch=100,
                    epochs=30,
                    validation_data=validation_generator,
                    validation_steps=50,
                    callbacks=callback_list)

# Show the tensorboard in the Linux terminal 
# -tensorboard --logdir logs/fit

model.save('inception_v4_weights_tf_cats_and_dogs_small.h5')

# Draw the general curves 
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


def smooth_curve(points, factor=0.8):
    # Define the smooth curve function  
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# Plot the smoothed curve 
plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# Evaluate the fine-tuned model on the test data
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(299, 299),
        batch_size=20,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)

# Release the GPU Memory
cuda.select_device(0)
cuda.close()
