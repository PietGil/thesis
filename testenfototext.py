from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2

im=cv2.imread('frame180.jpg')
eerste="test type"
tweede= "test data"
font= font= cv2.FONT_HERSHEY_DUPLEX
textbar='Last Barcode -> Type : %s ->Data : %s'  %(eerste,tweede)
cv2.putText(im,textbar,(100,100),font,1,(0,255,0),1,cv2.LINE_AA)
cv2.putText(im,textbar,(100,125),font,1,(0,255,0),1,cv2.LINE_AA)
cv2.imshow('opl',im)
cv2.waitKey()