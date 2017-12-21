import cv2
import numpy as np
cam = cv2.VideoCapture(0)
lower = np.array([0,48,80],dtype='uint8')
upper = np.array([20,255,255],dtype='uint8')

while  True:
    _,mat = cam.read()
    cv2.waitKey(1)
    cv2.imshow('1',mat)
    frame = mat.copy()
    cv2.cvtColor(mat,cv2.COLOR_BGR2HSV,dst=mat)
    mat = cv2.inRange(mat,lower,upper)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    mat = cv2.erode(mat,kernel,iterations=2)
    mat = cv2.dilate(mat,kernel,iterations=2)

    mat = cv2.GaussianBlur(mat,(3,3),0)
    mat = cv2.bitwise_and(frame,frame,mask=mat)

    frame = mat.copy()
    mat = cv2.cvtColor(mat,cv2.COLOR_BGR2GRAY)

    _,contours,hiearachy = cv2.findContours(mat,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame,contours,0,(0,0,255),3)
    cv2.waitKey(1)
    cv2.imshow('k',frame)
