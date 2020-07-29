# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:18:05 2019

@author: HP 250 G5
"""

import cv2

img=cv2.imread('ColorGameImg.jpg')
img = cv2.line(img, (100,50), (120,70), (0,0,0), 2)
img=cv2.line(img,(100,70),(120,50),(0,0,0), 2)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
