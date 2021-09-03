import numpy as np
import cv2
import keyboard 

def to3d(img,img2):

    img[:, :, 2] = 0
    img2[ :,  :, 0] = 0
    img2[ :, :,  1] = 0
    img3 = img + img2
    cv2.imshow("test", img3)

def sum(img,img2):
    img3 = img + img2
    cv2.imshow("mm", img3)
