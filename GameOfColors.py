# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:34:19 2019

@author: HP 250 G5
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:13:17 2019

@author: HP 250 G5
"""

import numpy as np
import cv2

# Load the models built in the previous steps
colors=("Red","Blue","Green")
#redLower = (0, 65, 100) redUpper = (20, 250, 250)
#(36, 25, 25), (70, 255,255)
ranges={"Red":[[170,100,0],[180,255,255]],"Blue":[[110,50,50],[130,255,255]],"Green":[[40,100,50],[75,255,255]]}

d_color="Green"

blueLower = np.array([110,50,50])
blueUpper = np.array([130,255,255])
# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)
image = cv2.imread('ColorGameImg.jpg')
image = cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)

mapping={}
#Create masks for each color
for color in colors:
    lower_range=np.asarray(ranges[color][0])
    upper_range=np.asarray(ranges[color][1])
    hsv_im=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask_im=cv2.inRange(hsv_im,lower_range,upper_range)
    ret, thresh = cv2.threshold(mask_im, 127, 255, 0)
    thresh=cv2.dilate(thresh,kernel,iterations=1)
    _,contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    mapping[color]=contours    



camera = cv2.VideoCapture(0)

#dist = cv2.pointPolygonTest(cnt,(50,50),True)

# Keep looping
while True:
    # Grab the frame
        (grabbed, frame) = camera.read()
        if not grabbed:
            break
        
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check to see if we have reached the end of the video 
        # (useful when input is a video file not a live video stream)
        if not grabbed:
            break
        
        
        
        
    # Determine which pixels fall within the blue boundaries and then blur the binary image
        blueMask = cv2.inRange(hsv, blueLower, blueUpper)
        blueMask = cv2.erode(blueMask, kernel, iterations=2)
        blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
        blueMask = cv2.dilate(blueMask, kernel, iterations=1)
    
        # Find contours (bottle cap in my case) in the image
        (_,cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center=None
        
        
        
        
    # Check to see if any contours were found
        	# Sort the contours and find the largest one -- we
        	# will assume this contour correspondes to the area of the bottle cap
        if len(cnts)>0:    
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
                # Get the radius of the enclosing circle around the found contour
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            center=(int(x),int(y))
                # Draw the circle around the contour
                #cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                # Get the moments to calculate the center of the contour (in this case Circle)
            
            for color in mapping.keys():
                flag=True
                for cnt in mapping[color]:
                    dist= cv2.pointPolygonTest(cnt,center,True)
                    if dist>=0:
                        M = cv2.moments(cnt)
                        centerX =int(M['m10'] / M['m00'])
                        centerY =int(M['m01'] / M['m00'])
                        if color==d_color:
                            image = cv2.line(image, (centerX-5,centerY-10) , (centerX+5,centerY+10), (0,0,0), 2)
                            image=cv2.line(image,(centerX+5,centerY+10),(centerX+25,centerY-25),(0,0,0), 2)  

                        else: 
                            image = cv2.line(image, (centerX,centerY) , (centerX+20,centerY+20), (0,0,0), 2)
                            image=cv2.line(image,(centerX,centerY+20),(centerX+20,centerY),(0,0,0), 2)  
                        flag=False
                        break
                if flag==False: break
                
                
            image_up=image.copy()    
            cv2.circle(image_up,center,5,(0,255,255),-1)
            cv2.imshow("Frame",image_up)
                
        else: cv2.imshow("Frame",image)
            
        #cv2.imshow("o_frame",frame)    
            
        
    # If the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
