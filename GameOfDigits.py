# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:13:17 2019

@author: HP 250 G5
"""

from keras.models import load_model
import numpy as np
import cv2
import random
from interface import Interface

def game_of_digits(color_point,level):
        
    level=int(level)
    inf_ob=Interface('gameofdigits')
    max_level=inf_ob.getMaxLevel()
    if level>max_level:
        temp = np.ones((480, 640,3), dtype=np.uint8)
        temp*=255
        cv2.putText(temp,"You have completed all levels of this game",(50,220),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0,),2,cv2.LINE_AA)  
        cv2.imshow('Message',temp)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        return

    blueLower1=None
    blueUpper1=None
    blueLower2=None
    blueUpper2=None

    if color_point=='Blue':
        blueLower1 = np.array([100, 60, 60])
        blueUpper1 = np.array([140, 255, 255])
        blueLower2 = np.array([100, 60, 60])
        blueUpper2 = np.array([140, 255, 255])
        
    elif color_point=='Red':
        blueLower1 = np.array([170,70,50])
        blueUpper1 = np.array([180,255,255])
        blueLower2 = np.array([0,70,50])
        blueUpper2 = np.array([10,255,255])
        
    else: #[40,100,50],[75,255,255]
        blueLower1 = np.array([36, 25, 25])
        blueUpper1 = np.array([70, 255, 255])
        blueLower2 = np.array([36, 25, 25])
        blueUpper2 = np.array([70, 255, 255])
        
    # Load the models built in the previous steps
    cnn_model = load_model('new_digit.h5')
    
    # Letters lookup
    min_area=1200
    max_area=10000
    
            
    kernel = np.ones((5, 5), np.uint8)
            
    camera = cv2.VideoCapture(0)
    
    
    
    def isLevelUp(scores):
        total=0
        for score in scores:
            total=total+score
            
        avg=total/(len(scores)*100)
        if avg>=0.75: 
            inf_ob.update_scores(level,avg)
            return True
        else: return False
    
    def check(img,image,cnt):
        n_image=image.copy()
        n_img=img.copy()
        n_image = cv2.cvtColor(n_image, cv2.COLOR_BGR2GRAY)
        n_img = cv2.cvtColor(n_img, cv2.COLOR_BGR2GRAY)
        x, y, w, h = cv2.boundingRect(cnt)
        img1 =n_img[y+10:y + h - 10, x+10:x + w - 10]
        img2 =n_image[y+10:y + h - 10, x+10:x + w - 10]
        img1=np.asarray(img1)
        img2=np.asarray(img2)
        return np.array_equal(img1,img2)
    
    def preprocess(image,mapping):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray=cv2.bitwise_not(gray)
        ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for l in range(10):
            mapping[l]=[]
            
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            alphabet = gray[y+10:y + h - 10, x+10:x + w - 10]
            #cv2.imshow("Cropped Letter",alphabet)
            #cv2.waitKey(0)
            newImage = cv2.resize(alphabet, (28, 28),interpolation = cv2.INTER_CUBIC)
            newImage = np.array(newImage)
            newImage = newImage.astype('float32')/255
            prediction = cnn_model.predict(newImage.reshape(1,28,28,1))
            prediction = np.argmax(prediction)
            #cv2.putText(image,str(letters[int(prediction)+1]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)    
            predicted_digit=int(prediction)
            mapping[predicted_digit].append(cnt)
    #    return len(contours)
    
    
    def getContour(cnts):
        cnt=None
        cur_area=0
        for ct in cnts:
            area=float(cv2.contourArea(ct))
            if area>=min_area and area<=max_area and area>cur_area:
                cur_area=area
                cnt=ct
        return cnt
    
    

    
    br=True
    
    while level<=max_level:
        temp = np.ones((480, 640,3), dtype=np.uint8)
        temp*=255
        cv2.putText(temp,"Level"+str(level),(200,220),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0,),2,cv2.LINE_AA)
        temp_2=cv2.resize(temp,(1400,750))
        cv2.imshow('Frame',temp_2)
        cv2.waitKey(5000)
        img = cv2.imread('images/DigitGame/DigitGameImg'+str(level)+'.jpg')
        img = cv2.resize(img, (640,480), interpolation = cv2.INTER_AREA)
        mapping={}
        preprocess(img,mapping)
        scores=[]
        dist_digit=[]
        for digit in range(10):
            if len(mapping[digit])==0: continue
            dist_digit.append(digit)
            
        random.shuffle(dist_digit)
        
        for d_digit in dist_digit:
            temp = np.ones((480, 640,3), dtype=np.uint8)
            temp*=255
            cv2.putText(temp,"Pick all '"+str(d_digit)+"'",(200,220),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)
            temp_2=cv2.resize(temp,(1400,750))
            cv2.imshow('Frame',temp_2)
            cv2.waitKey(5000)
            count=len(mapping[d_digit])
            image=img.copy()
            cv2.putText(image,"Pick all '"+str(d_digit)+"'",(270,460),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0,),1,cv2.LINE_AA)
            cur_score=0
            tcount=0
            while count>0:
                # Grab the frame
                    (grabbed, frame) = camera.read()
                    if not grabbed:
                       continue
                    
                    frame = cv2.flip(frame, 1)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    blueMask1 = cv2.inRange(hsv, blueLower1, blueUpper1)
                    blueMask2 = cv2.inRange(hsv, blueLower2, blueUpper2)
                    blueMask=blueMask1|blueMask2
                    blueMask = cv2.erode(blueMask, kernel, iterations=2)
                    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
                    blueMask = cv2.dilate(blueMask, kernel, iterations=1)
                
                    # Find contours (bottle cap in my case) in the image
                    (cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    center=None
                    
                    
                    
                    
                # Check to see if any contours were found
                    	# Sort the contours and find the largest one -- we
                    	# will assume this contour correspondes to the area of the bottle cap
                    if len(cnts)>0:    
                        cnt =getContour(cnts)
                        if cnt is not None:
                            # Get the radius of the enclosing circle around the found contour
                            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                            center=(int(x),int(y))
                                # Draw the circle around the contour
                            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                                # Get the moments to calculate the center of the contour (in this case Circle)
                            
                            for digit in range(10):
                                flag=True
                                for cnt in mapping[digit]:
                                    dist= cv2.pointPolygonTest(cnt,center,True)
                                    if dist>=0:
                                        if digit==d_digit:
                                            if check(img,image,cnt): 
                                                count=count-1
                                                tcount=tcount+1
                                                cur_score=cur_score+20
                                            cv2.drawContours(image, [cnt], 0, (0, 255, 0), -1)
                                        else: 
                                            if check(img,image,cnt): 
                                                tcount=tcount+1
                                                cur_score=cur_score-5
                                            cv2.drawContours(image, [cnt], 0, (0, 0, 255), -1)
                                        flag=False
                                        break
                                if flag==False: break
                        
                            
                            image_up=image.copy()    
                            cv2.circle(image_up,center,5,(0,255,255),-1)
                            image_up_2=cv2.resize(image_up,(1400,750))
                            cv2.imshow("Frame",image_up_2)
                            if count==0: cv2.waitKey(1000)
                            
                    else: 
                        image_2=cv2.resize(image,(1400,750))
                        cv2.imshow("Frame",image_2)
                        
                    #cv2.imshow("o_frame",frame)    
                        
                    
                # If the 'q' key is pressed, stop the loop
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        br=False
                        break
            
            if not br: break
            x=float(cur_score)*100/(tcount*20)
            if x<0: x=0
            temp = np.ones((480, 640,3), dtype=np.uint8)
            temp*=255
            cv2.putText(temp,"Your Score: "+str(round(x,2))+"%",(200,220),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)
            temp_2=cv2.resize(temp,(1400,750))
            cv2.imshow('Frame',temp_2)
            cv2.waitKey(5000)
            scores.append(x)
        if not br: break
        if isLevelUp(scores): 
            temp = np.ones((480, 640,3), dtype=np.uint8)
            temp*=255
            if level<max_level: cv2.putText(temp,"Congrats, Level UP!!",(200,220),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)
            else: cv2.putText(temp,"Congrats, all levels completed",(90,220),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)  
            temp_2=cv2.resize(temp,(1400,750))
            cv2.imshow('Frame',temp_2)
            cv2.waitKey(5000)
            level=level+1
        else:
            temp = np.ones((480, 640,3), dtype=np.uint8)
            temp*=255
            cv2.putText(temp,"Try again Level "+str(level)+"!!",(200,220),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)
            temp_2=cv2.resize(temp,(1400,750))
            cv2.imshow('Frame',temp_2)
            cv2.waitKey(5000)
        
            # Cleanup the camera and close any open windows
            
    camera.release()
    cv2.destroyAllWindows()
