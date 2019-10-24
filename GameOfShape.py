# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:13:17 2019

@author: HP 250 G5
"""

import numpy as np
import cv2

# Load the models built in the previous steps
shapes=("Rectangle","Square","Triangle","Star","Circle")

d_shape=shapes[2]

blueLower = np.array([110,50,50])
blueUpper = np.array([130,255,255])
# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)
image = cv2.imread('someshapes.jpg')
image = cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, 1)

_,contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

mapping={}

for cnt in contours:
    # Get approximate polygons
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True),True)
    shape_name=None
    
    if len(approx) == 3:
        shape_name = shapes[2]
    
    elif len(approx) == 4:
        x,y,w,h = cv2.boundingRect(cnt)
        # Check to see if 4-side polygon is square or rectangle
        # cv2.boundingRect returns the top left and then width and 
        if abs(w-h) <= 5:
            shape_name = shapes[1]
        else:
            shape_name = shapes[0]
            
    elif len(approx) == 10:
        shape_name = shapes[3]
        
    elif len(approx) >= 15:
        shape_name = shapes[4]
    
    mapping[shape_name]=cnt    


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
            
            for shape in mapping.keys():
                dist= cv2.pointPolygonTest(mapping[shape],center,True)
                if dist>=0:
                    if shape==d_shape:
                        cv2.drawContours(image, [mapping[shape]], 0, (0, 255, 0), -1)
                    else: 
                        cv2.drawContours(image, [mapping[shape]], 0, (0, 0, 255), -1)
                    break    
                
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
