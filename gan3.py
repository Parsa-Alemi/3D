from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import Sequential
import keras_metrics
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, Dropout,Conv2DTranspose, GlobalMaxPooling2D,UpSampling2D,MaxPooling2D, Concatenate
import tensorflow as tf
from keras.optimizers import SGD
import functools
import numpy as np
from itertools import product
from functools import partial
import keras.applications
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os
import random
from keras import Sequential
import keras_metrics
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers import Dense, Input , Conv2D, MaxPool2D, Flatten, Activation, Dropout, GlobalMaxPooling2D
import tensorflow as tf
from keras.optimizers import Adam,Adadelta,RMSprop,Adamax
import functools
import numpy as np
from itertools import product
from functools import partial
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import matplotlib.pyplot as plt
import numpy as np

rast = np.load("x256.npy")
chap = np.load("y256.npy")

#rast = rast[:10]
#chap = chap[:10]
img_rows = 256
img_cols = 256
channels = 3
img_shape = (img_rows, img_cols, channels)
smooth = 1.
z_dim = (256,256,3)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)


def generator(inputs):
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool5)
    # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(convdeep)
    
    # upmid = merge([Convolution2D(512, 2, 2, border_mode='same')(UpSampling2D(size=(2, 2))(convdeep)), conv5], mode='concat', concat_axis=1)
    # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(upmid)
    # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(convmid)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Convolution2D(256, 2, 2,activation='relu', border_mode='same')(up6)
    up6 = Concatenate(axis=- 1)([up6, conv4])
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Convolution2D(128, 2, 2,activation='relu', border_mode='same')(up7)
    up7 = Concatenate(axis=- 1)([up7, conv3])
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Convolution2D(64, 2, 2,activation='relu', border_mode='same')(up8)
    up8 = Concatenate(axis=- 1)([up8, conv2])
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)


    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Convolution2D(32, 2, 2,activation='relu', border_mode='same')(up9)
    up9 = Concatenate(axis=- 1)([up9, conv1])
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(3, 1, 1, activation='tanh')(conv9)

    model = Model(input=inputs, output=conv10)
    print("Gen")
    model.summary()
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def discriminator(img_shape):

    model = Sequential()
    model.add(Conv2D(8, kernel_size=3, strides=2, 
                             input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(16, kernel_size=3, strides=2, 
                             input_shape=img_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, kernel_size=3, strides=2, 
                             input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=3, strides=2, 
                             input_shape=img_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(128, kernel_size=3, strides=2, 
                             input_shape=img_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=img_shape)
    prediction = model(img)
    print("dis")
    model.summary()
    
    return Model(img, prediction)

discriminator = discriminator((256,256,6))
discriminator.compile(loss='binary_crossentropy', 
                      optimizer=Adam(), metrics=['accuracy'])
z = Input((256,256,3))
# Build the Generator
generator = generator(z)
#generator = keras.models.load_model("Models/Generator/a2-1010.h5")
# Generated image to be used as input

img = generator(z)

# Keep Discriminator’s parameters constant during Generator training
discriminator.trainable = False

img_cat = concatenate([z, img], axis=3)

# The Discriminator’s prediction
prediction = discriminator(img_cat)
print(img_cat)

# Combined GAN model to train the Generator
combined = Model(z, prediction)
combined.compile(loss='binary_crossentropy', optimizer=Adam())
losses = []
accuracies = []

def train(iterations, batch_size, sample_interval):
    


    # Labels for real and fake examples
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        
        # -------------------------
        #  Train the Discriminator
        # -------------------------

        # Select a random batch of real images
        idx = np.random.randint(0, rast.shape[0], batch_size)
        imgs = rast[idx]
        #imgs = tf.convert_to_tensor(imgs,np.float32)
        true_imgs = chap[idx]
        #true_imgs = tf.convert_to_tensor(true_imgs,np.float32)
        #real_input = concatenate([true_imgs, imgs], axis=3)
        real_input = np.concatenate((true_imgs, imgs), axis=3)


        idx = np.random.randint(0, chap.shape[0], batch_size)
        z = chap[idx]
        gen_imgs = generator.predict(z)
        #z = tf.convert_to_tensor(z,np.float32)
        #gen_imgs = tf.convert_to_tensor(gen_imgs,np.float32)
        #fake_input = concatenate([z, gen_imgs], axis=3)
        fake_input = np.concatenate((z, gen_imgs), axis=3)

        #dis hardcore
        rnd = np.random.randint(batch_size, chap.shape[0], 1)
        m = np.random.randint(0, chap.shape[0], batch_size)
        n = np.random.randint(0, chap.shape[0], batch_size)
        z2 = chap[np.array(m)]
        imgs2 = rast[np.array(n)]
        fake_input2 = np.concatenate((z2, imgs2), axis=3)
        # Discriminator loss
        d_loss_real = discriminator.train_on_batch(real_input, real)
        d_loss_fake = discriminator.train_on_batch(fake_input, fake)
        d_loss_fake2 = discriminator.train_on_batch(fake_input2, fake)

        d_loss =np.add(np.add(d_loss_real, d_loss_fake), d_loss_fake2) / 3



        # ---------------------
        #  Train the Generator
        # ---------------------

        # Generate a batch of fake images
        idx= np.random.randint(0, chap.shape[0], batch_size)
        z = chap[idx]
        gen_imgs = generator.predict(z)

        # Generator loss
        g_loss = combined.train_on_batch(z, real)

        if iteration % sample_interval == 0:
            
            # Output training progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % 
                         (iteration, d_loss[0], 100*d_loss[1], g_loss))
            generator.save("models/Generator/a5-" + str(iteration) + ".h5")
            discriminator.save("models/Discriminator/a5-" + str(iteration) + ".h5")
            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss[0], g_loss))
            accuracies.append(100*d_loss[1])

iterations = 30000
batch_size = 16
sample_interval = 10

# Train the GAN for the specified number of iterations
train(iterations, batch_size, sample_interval)
