import scipy
from glob import glob
import numpy as np
import os
import random

from keras import Sequential
import keras_metrics
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, Dropout,Conv2DTranspose, GlobalMaxPooling2D,UpSampling2D,MaxPooling2D
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
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, Dropout, GlobalMaxPooling2D
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



def load_data(dir):
    X = []
    y = []

    files = os.listdir(dir)

    random.shuffle(files)
    for file in files:
        if file[-3:] == 'jpg' and file[0] != 'y':
            img = cv2.imread(os.path.join(dir, file))
            img2 = cv2.imread(os.path.join(dir, "y_" + file))


            img = img[13:240, :]
            img2 = img2[40:200, :]

            
            img = cv2.resize(img, (256, 256))
            img2 = cv2.resize(img2, (256, 256))
            img = img.reshape((256,256,3,))
            img2 = img2.reshape((256,256,3,))
            #img = img.reshape((7500,))

            
            img = (img - 127) / 127
            img2 = (img2 - 127) / 127

            X.append(img)
            y.append(img2)

    return np.array(X), np.array(y)
    
x_train , y_train = load_data("dataset/train")
np.save("x256.npy" , x_train)
np.save("y256.npy" , y_train)
