import cv2
import numpy as np
lower_range = np.array([155,25,0])
upper_range = np.array([179,255,255])
video_capture=cv2.VideoCapture(0)
while True:
    ret,frame=video_capture.read()
    if ret is True:
        hsv_im=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask_im=cv2.inRange(hsv_im,lower_range,upper_range)
        rs=cv2.bitwise_and(frame,frame,mask=mask_im)
        cv2.imshow('Mask',mask_im)
        cv2.imshow('Frame',frame)
        cv2.imshow('Result',rs)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
