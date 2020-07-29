# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:14:01 2019

@author: HP 250 G5
"""

from keras.models import load_model
from collections import deque
from num2words import convert
import numpy as np
import cv2

def game_of_num2words(color_point,color_pen):
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
        blueLower2 = np.array([36, 25, 25])
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

    model=load_model('new_digit.h5')
    
    def find_number(frame,model):
        num=0
        k=0
        flag=True
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray=cv2.bitwise_not(gray)
        ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        (cnts, heirarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            	cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts)==0: flag=False
        cnts=sorted(cnts,key=lambda ctr: cv2.boundingRect(ctr)[0], reverse=True)
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            img = gray[y-20:y + h + 20, x-20:x + w + 20]
            newImage = cv2.resize(img, (28, 28), cv2.INTER_CUBIC)
            newImage = np.array(newImage)
            newImage = newImage.astype('float32')/255
            newImage=newImage.reshape(1,28,28,1)
            prediction = model.predict(newImage)
            prediction = np.argmax(prediction)
            num=int(prediction)*pow(10,k)+num
            k=k+1
            
        return flag,str(num)
    
    def showTimer():
        for i in range(10,-1,-1):
            temp = np.ones((480, 640,3), dtype=np.uint8)
            temp*=255
            cv2.putText(temp,"Guess answer in your mind!!",(100,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)    
            cv2.putText(temp,str(i),(300,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)
            cv2.imshow('WhiteBoard',temp)
            cv2.waitKey(1000)
            
    # Define the upper and lower boundaries for a color to be considered "Blue"
    min_area=1200
    max_area=10000
    
    
    # Define a 5x5 kernel for erosion and dilation
    kernel = np.ones((5, 5), np.uint8)
    
    number = np.ones((480, 640,3), dtype=np.uint8)
    number*=255
    
    
    
    points = deque(maxlen=1024)
    
    def getContour(cnts):
        cnt=None
        cur_area=0
        for ct in cnts:
            area=float(cv2.contourArea(ct))
            if area>=min_area and area<=max_area and area>cur_area:
                cur_area=area
                cnt=ct
        return cnt
    
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
            ( cnts_blue, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
            	cv2.CHAIN_APPROX_SIMPLE)
            center = None
            ( cnts_red, _) = cv2.findContours(redMask.copy(), cv2.RETR_EXTERNAL,
            	cv2.CHAIN_APPROX_SIMPLE)
            # Check to see if any contours were found
            #im_cpy=whiteboard.copy()
            whiteboard = np.ones((480, 640,3), dtype=np.uint8)
            whiteboard*=255
                
                
            cv2.putText(whiteboard,'Write a number ( 1 to 99999 ):',(50,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)
            cv2.rectangle(whiteboard, (50,80), (590,300), (0,255,0), 2)
            cnt1=np.asarray([[(50,80)],[(50,300)],[(590,300)],[(590,80)]])
            cv2.rectangle(whiteboard,(120,350), (270,400), (0,255,0), 2)
            cnt2=np.asarray([[(120,350)],[(120,400)],[(270,400)],[(270,350)]])
            cv2.putText(whiteboard,'Done',(155,385),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)
            cv2.rectangle(whiteboard, (370,350), (520,400), (0,255,0), 2)
            cnt3=np.asarray([[(370,350)],[(370,400)],[(520,400)],[(520,350)]])
            cv2.putText(whiteboard,'Clear',(405,385),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)
            
    
            
            
            
            	# Sort the contours and find the largest one -- we
            cnt = getContour(cnts_blue)
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
                         #center =(int(x),int(y))
                         M = cv2.moments(cnt)
                         center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                         cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                         cv2.circle(whiteboard, center,5,(0,255,0),-1)
                         
                         dist= cv2.pointPolygonTest(cnt2,center,True)
                         if dist>=0 and count==0:
                             #time.sleep(1)
                             number=number[90:290,60:580]
                             check,n=find_number(number,model)
                             if check: 
                                 showTimer()
                                 convert(n)
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
    
        
            number=whiteboard
            # Show the frame
            cv2.imshow("WhiteBoard", whiteboard)
            
            
            if(count>0): count=count-1
        except:
            pass
        # If the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    camera.release()
    cv2.destroyAllWindows()
