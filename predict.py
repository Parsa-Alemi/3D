from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, Dropout, GlobalMaxPooling2D,UpSampling2D,MaxPooling2D
import tensorflow as tf
from keras.optimizers import SGD
import functools
import numpy as np
from itertools import product
from functools import partial
import keras.models
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os
import random
from PIL import Image

smooth = 1.
X = []

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

        
#model = keras.models.load_model("Models/Generator/a2-1010.h5")
model = keras.models.load_model("Models/Generator/a3-1360.h5")
#model = keras.models.load_model("a10-960.h5", custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef': dice_coef})
    
img = cv2.imread("dataset/train/76.jpg")


img = cv2.resize(img, (256, 256))
cv2.imshow("tt1",img)
img = (img - 127) / 127
img = cv2.resize(img, (256, 256))
img = img.reshape((1, 256, 256, 3))



pred = model.predict(img)
print(pred.shape)
+
img2 = pred[0]



img2 = (img2 * 127) + 127
print(img2)
img2 = cv2.resize(img2, (256, 256))


cv2.imshow("tt2", img2)
cv2.waitKey()
cv2.destroyAllWindows()

