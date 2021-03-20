__author__ = "Damian Andrysiak"
__license__ = "Feel free to copy"

import tensorflow as tf
from tensorflow import keras
from keras import layers

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3    #RGB or BGR


# Build the model

# Input 
input = layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255.0)(input)


# First Level
Conv1 = layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
Conv1 = layers.Dropout(0.1)(Conv1)
Conv1 = layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(Conv1)
Pool1 = layers.MaxPooling2D((2,2))(Conv1)


# Second Level
Conv2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Pool1)
Conv2 = layers.Dropout(0.1)(Conv2)
Conv2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Conv2)
Pool2 = layers.MaxPooling2D((2, 2))(Conv2)
 

# Third Level
Conv3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Pool2)
Conv3 = layers.Dropout(0.2)(Conv3)
Conv3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Conv3)
Pool3 = layers.MaxPooling2D((2, 2))(Conv3)


# Fourth Level
Conv4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Pool3)
Conv4 = layers.Dropout(0.2)(Conv4)
Conv4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Conv4)
Pool4 = layers.MaxPooling2D(pool_size=(2, 2))(Conv4)
 
# Sixth Level
Conv5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Pool4)
Conv5 = layers.Dropout(0.3)(Conv5)
Conv5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Conv5)


# Seventh Level
Conv_Transpose6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(Conv5)
Conv_Transpose6 = layers.concatenate([Conv_Transpose6, Conv4])
Conv6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Conv_Transpose6)
Conv6 = layers.Dropout(0.2)(Conv6)
Conv6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Conv6)
 

# Eighth Level
Conv_Transpose7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(Conv6)
Conv_Transpose7 = layers.concatenate([Conv_Transpose7, Conv3])
Conv7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Conv_Transpose7)
Conv7 = layers.Dropout(0.2)(Conv7)
Conv7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Conv7)
 

# Nineth Level
Conv_Transpose8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(Conv7)
Conv_Transpose8 = layers.concatenate([Conv_Transpose8, Conv2])
Conv8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Conv_Transpose8)
Conv8 = layers.Dropout(0.1)(Conv8)
Conv8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Conv8)
 

# Tenth Level
Conv_Transpose9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(Conv8)
Conv_Transpose9 = layers.concatenate([Conv_Transpose9, Conv1], axis=3)
Conv9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Conv_Transpose9)
Conv9 = layers.Dropout(0.1)(Conv9)
Conv9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(Conv9)
 
# Eleventh Level
output = layers.Conv2D(1, (1, 1), activation='sigmoid')(Conv9)
 

# Model summary
model = tf.keras.Model(inputs=[input], outputs=[output])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print()
print()
print()
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################


# Lets test the model!
import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


import matplotlib.pyplot as plt

# Display a sample element

#elem = 100
#plt.imshow(x_train[elem])
#plt.show()

#print("Label for element No.", elem,":", y_train[elem])


t = tf.constant([[1, 2, 3], [4, 5, 6]])
paddings = tf.constant([[1, 1,], [3, 3]])

print(tf.pad(t, paddings, "CONSTANT"))





# Reshape the images
#print(x_train.shape)
#print(x_test.shape)

#x_train = x_train.reshape((-1, IMG_WIDTH*IMG_HEIGHT))
#x_test = x_test.reshape((-1, IMG_WIDTH*IMG_HEIGHT))

#print(x_train.shape)
#print(x_test.shape)

#print(x_train.min(), "-", x_train.max())