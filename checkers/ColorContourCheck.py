# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:58:44 2019

@author: HP 250 G5
"""

import cv2
import numpy as np

img=cv2.imread('ColorGameImg.jpg')
#170, 100, 0), (180, 255, 255)
#[36,25,25],[70,255,255]

lower_range = np.array([40,100,50])
upper_range = np.array([75,255,255])
kernel = np.ones((5, 5), np.uint8)
hsv_im=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
mask_im=cv2.inRange(hsv_im,lower_range,upper_range)
#mask_im = cv2.erode(mask_im, kernel, iterations=2)
#mask_im = cv2.morphologyEx(mask_im, cv2.MORPH_OPEN, kernel)
#mask_im = cv2.dilate(mask_im, kernel, iterations=1)
ret, thresh = cv2.threshold(mask_im, 127, 255, 0)
thresh=cv2.dilate(thresh,kernel,iterations=1)
_,contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 5)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(len(contours))

