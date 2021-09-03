import numpy as np
import cv2
import threading as th
import subprocess as sp
import io

cv2.waitKey(1000)

#Video capture requirments
#cap = cv2.VideoCapture('')
#cap2 = cv2.VideoCapture('')

i = 0
file=open("i.txt")
i=file.read()
i=int(i)
with io.open("i.txt", "w", encoding="utf-8") as f:
    f.write(str(i+1))

def take_image1():
    global cap, i
    ret, frame = cap.read()
    cv2.imwrite("dataset/train/x/"+str(i) + '.jpg',frame)
    cv2.imshow('image1', frame)
    cv2.waitKey(1000)

def take_image2():
    global cap2, i
    ret, frame = cap2.read()
    cv2.imwrite("dataset/train/y/y_"+str(i) + '.jpg',frame)
    cv2.imshow('image2', frame)
    cv2.waitKey(1000)

print("Ready!")
ret, frame = cap.read()
cv2.imwrite("dataset/train/x/"+str(i) + '.jpg',frame)
cv2.imshow('image1', frame)
ret, frame2 = cap2.read()
cv2.imwrite("dataset/train/y/y_"+str(i) + '.jpg',frame2)
cv2.imshow('image2', frame2)
cv2.waitKey(1000)
#th1 = th.Thread(target=take_image1)
#th2 = th.Thread(target=take_image2)
#th1.start()
#th2.start()
#th1.join()
#th2.join()


print("shit")
exit()
sp.call("python livecamera.py")

while(1):
    th1.start()
    th2.start()
    th1.join()
    th2.join()
    print(i)
    i += 1
    cv2.waitKey(3000)
        
cap.release()
cv2.destroyAllWindows()
