
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:41:25 2019

@author: HP 250 G5
"""

from collections import deque
import numpy as np
import cv2
import os
import re
import pandas as pd
# Define the upper and lower boundaries for a color to be considered "Blue"'#
def eSign(color_point,color_pen):
    min_area=1200
    max_area=10000
    
    blueLower1=None
    blueUpper1=None
    blueLower2=None
    blueUpper2=None
    redLower1=None
    redUpper1=None
    redLower2=None
    redUpper2=None
    
    if color_pen=='Blue':
        blueLower1 = np.array([100, 60, 60])
        blueUpper1 = np.array([140, 255, 255])
        blueLower2 = np.array([100, 60, 60])
        blueUpper2 = np.array([140, 255, 255])
        
    elif color_pen=='Red':
        blueLower1 = np.array([170,70,50])
        blueUpper1 = np.array([180,255,255])
        blueLower2 = np.array([0,70,50])
        blueUpper2 = np.array([10,255,255])
        
    else: #[40,100,50],[75,255,255]
        blueLower1 = np.array([36, 25, 25])
        blueUpper1 = np.array([70, 255, 255])
        blueLower2 = np.array([36, 100, 50])
        blueUpper2 = np.array([70, 255, 255])
    
    if color_point=='Blue':
        redLower1 = np.array([100, 60, 60])
        redUpper1 = np.array([140, 255, 255])
        redLower2 = np.array([100, 60, 60])
        redUpper2 = np.array([140, 255, 255])
        
    elif color_point=='Red':
        redLower1 = np.array([170,70,50])
        redUpper1 = np.array([180,255,255])
        redLower2 = np.array([0,70,50])
        redUpper2 = np.array([10,255,255])
        
    else: #[40,100,50],[75,255,255]
        redLower1 = np.array([36, 25, 25])
        redUpper1 = np.array([70, 255, 255])
        redLower2 = np.array([36, 25, 25])
        redUpper2 = np.array([70, 255, 255])
    # Define a 5x5 kernel for erosion and dilation
    kernel = np.ones((5, 5), np.uint8)
    
    signature = np.ones((480, 640,3), dtype=np.uint8)
    signature*=255
    
    def save_signature(signature):
        df=pd.read_csv('configure.csv')
        uid=int(df.iloc[0][0])
        df=pd.read_csv('login.csv')
        name=str(df.iloc[uid][1]).capitalize()+str(df.iloc[uid][2]).capitalize()
        temp=os.path.join('signatures',name)
        if not os.path.exists('signatures'):
            os.mkdir('signatures')
        if not os.path.exists(temp):
            os.mkdir(temp)
        max_num=0
        for root, dirs, files in os.walk(temp):
            for filename in files:
                #if not filename.contains('.jpg'): continue
                x=re.findall('[0-9]',filename)
                if x:
                    num=''
                    for e in x:
                        num=num+str(e)
                    num=int(num)
                    if num>max_num: max_num=num
                    
        max_num=max_num+1
        cv2.imwrite(os.path.join(temp,'signature'+str(max_num)+'.jpg'),signature)
    
    def getContour(cnts):
        cnt=None
        cur_area=0
        for ct in cnts:
            area=float(cv2.contourArea(ct))
            if area>=min_area and area<=max_area and area>cur_area:
                cur_area=area
                cnt=ct
        return cnt
    
    
    points = deque(maxlen=1024)
    # Load the video
    camera = cv2.VideoCapture(0)
    flag=1
    count=0
    cv2.waitKey(5000)
    while True:
        try:
            (grabbed, frame) = camera.read()
            if not grabbed:
                continue
            frame = cv2.flip(frame, 1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
            # Determine which pixels fall within the blue boundaries and then blur the binary image
            blueMask1 = cv2.inRange(hsv, blueLower1, blueUpper1)
            blueMask2 = cv2.inRange(hsv, blueLower2, blueUpper2)
            blueMask=blueMask1|blueMask2
            blueMask = cv2.erode(blueMask, kernel, iterations=2)
            blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
            blueMask = cv2.dilate(blueMask, kernel, iterations=1)
            
            redMask1 = cv2.inRange(hsv, redLower1, redUpper1)
            redMask2 = cv2.inRange(hsv, redLower2, redUpper2)
            redMask=redMask1|redMask2
            redMask = cv2.erode(redMask, kernel, iterations=2)
            redMask = cv2.morphologyEx(redMask, cv2.MORPH_OPEN, kernel)
            redMask = cv2.dilate(redMask, kernel, iterations=1)
        
            # Find contours (bottle cap in my case) in the image
            (cnts_blue, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
            	cv2.CHAIN_APPROX_SIMPLE)
            center = None
            (cnts_red, _) = cv2.findContours(redMask.copy(), cv2.RETR_EXTERNAL,
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
            
    
            
            
            #if len(cnts_blue) > 0:
            	# Sort the contours and find the largest one -- we
            cnt =getContour(cnts_blue)
            if cnt is not None:
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
                     cnt = getContour(cnts_red)
                     if cnt is not None:
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
                             #cv2.imwrite("images/signature.jpg",signature)
                             save_signature(signature)
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
                    #cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 2)
                    cv2.line(whiteboard, points[i - 1], points[i], (255, 0, 0), 2)
    
        
            signature=whiteboard
            # Show the frame
            cv2.imshow("eSign", whiteboard)
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