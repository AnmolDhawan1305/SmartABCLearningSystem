# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:06:08 2019

@author: HP 250 G5
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:49:48 2019

@author: HP 250 G5
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:41:25 2019

@author: HP 250 G5
"""

from collections import deque
import numpy as np
import cv2
import time

# Define the upper and lower boundaries for a color to be considered "Blue"
redLower = np.array([170,50,0])
redUpper = np.array([180,255,255])

blueLower = np.array([110,50,50])
blueUpper = np.array([130,255,255])


# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

signature = np.ones((480, 640,3), dtype=np.uint8)
signature*=255



points = deque(maxlen=1024)
# Load the video
camera = cv2.VideoCapture(0)
flag=1
count=0

while True:
    try:
        (grabbed, frame) = camera.read()
        if not grabbed:
            continue
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Determine which pixels fall within the blue boundaries and then blur the binary image
        blueMask = cv2.inRange(hsv, blueLower, blueUpper)
        blueMask = cv2.erode(blueMask, kernel, iterations=2)
        blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
        blueMask = cv2.dilate(blueMask, kernel, iterations=1)
        
        redMask = cv2.inRange(hsv, redLower, redUpper)
        redMask = cv2.erode(redMask, kernel, iterations=2)
        redMask = cv2.morphologyEx(redMask, cv2.MORPH_OPEN, kernel)
        redMask = cv2.dilate(redMask, kernel, iterations=1)
    
        # Find contours (bottle cap in my case) in the image
        (_, cnts_blue, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        center = None
        (_, cnts_red, _) = cv2.findContours(redMask.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        # Check to see if any contours were found
        #im_cpy=whiteboard.copy()
        whiteboard = np.ones((480, 640,3), dtype=np.uint8)
        whiteboard*=255
            
            
        cv2.putText(whiteboard,'Sign here:',(50,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)
        cv2.rectangle(whiteboard, (50,80), (590,300), (0,255,0), 2)
        cnt1=np.asarray([[(50,80)],[(50,300)],[(590,300)],[(590,80)]])
        cv2.rectangle(whiteboard,(120,350), (270,400), (0,255,0), 2)
        cnt2=np.asarray([[(120,350)],[(120,400)],[(270,400)],[(270,350)]])
        cv2.putText(whiteboard,'Save',(155,385),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)
        cv2.rectangle(whiteboard, (370,350), (520,400), (0,255,0), 2)
        cnt3=np.asarray([[(370,350)],[(370,400)],[(520,400)],[(520,350)]])
        cv2.putText(whiteboard,'Clear',(405,385),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)
        

        
        
        if len(cnts_blue) > 0:
        	# Sort the contours and find the largest one -- we
            cnt = sorted(cnts_blue, key = cv2.contourArea, reverse = True)[0]
            # Get the radius of the enclosing circle around the found contour
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            # Draw the circle around the contour
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            # Get the moments to calculate the center of the contour (in this case Circle)
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    
            dist= cv2.pointPolygonTest(cnt1,center,True)
            
            if dist>=0: 
                points.appendleft(center)
                flag=1
    
        else :
             if len(points) != 0:
                if(flag==1): 
                    points.appendleft((-1,-1))
                    flag=0
                    
             if len(cnts_red)>0:
                 cnt = sorted(cnts_red, key = cv2.contourArea, reverse = True)[0]
                    # Get the radius of the enclosing circle around the found contour
                 ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                    # Draw the circle around the contour
                 center =(int(x),int(y))
                 cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                 cv2.circle(whiteboard, center,5,(0,255,0),-1)
                 
                 dist= cv2.pointPolygonTest(cnt2,center,True)
                 if dist>=0 and count==0:
                     #time.sleep(1)
                     count=5
                     signature=signature[90:290,60:580]
                     cv2.imshow("signature",signature)
                     cv2.waitKey(0)
                     cv2.destroyWindow('signature')
                     cv2.imwrite("images/signature.jpg",signature)
                     points = deque(maxlen=1024)
                 
                 else:
                     dist= cv2.pointPolygonTest(cnt3,center,True)
                     if dist>=0:
                         points = deque(maxlen=1024)
             
    
        # Connect the points with a line
        for i in range(1, len(points)):
                if points[i - 1] is None or points[i] is None:
                        continue
                if points[i]==(-1,-1) or points[i-1]==(-1,-1):
                    continue
                cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 2)
                cv2.line(whiteboard, points[i - 1], points[i], (255, 0, 0), 2)

    
        signature=whiteboard
        # Show the frame
        cv2.imshow("WhiteBoard", whiteboard)
        #cv2.imshow("signatures",signature)
        #cv2.imshow("Frame",frame)
        #whiteboard.fill(255)
        if(count>0): count=count-1
    except:
        pass
    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
camera.release()
cv2.destroyAllWindows()