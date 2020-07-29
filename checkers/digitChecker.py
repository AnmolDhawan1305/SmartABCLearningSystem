# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 20:37:39 2019

@author: HP 250 G5
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:51:22 2019

@author: HP 250 G5
"""

from keras.models import load_model
import numpy as np
import cv2

# Load the models built in the previous steps
cnn_model = load_model('mnist_keras_cnn_model.h5')

# Letters lookup

image = cv2.imread('images/DigitGameImg.jpg')
image = cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray=cv2.bitwise_not(gray)
cv2.imshow("Original Image",image)
#ret, thresh = cv2.threshold(gray, 127, 255, 0)\
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("Working Image",thresh)
_,contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
print(len(contours))
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    alphabet = gray[y+5:y + h - 5, x+5:x + w - 5]
    cv2.imshow("Cropped digit",alphabet)
    cv2.waitKey(0)
    newImage = cv2.resize(alphabet, (28, 28),interpolation = cv2.INTER_AREA)
    newImage = np.array(newImage)
    #newImage = newImage.astype('float32')/255
    prediction = cnn_model.predict(newImage.reshape(1,28,28,1))[0]
    prediction = np.argmax(prediction)
    cv2.putText(image,str(int(prediction)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)    

cv2.imshow("Prediction",image)
cv2.waitKey(0)
cv2.destroyAllWindows()