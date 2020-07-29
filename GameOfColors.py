# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:39:48 2019

@author: HP 250 G5
"""




import numpy as np
import cv2
import random
from interface import Interface

def game_of_colors(color_point,level):
    # Letters lookup
    level=int(level)
    inf_ob=Interface('gameofcolors')
    max_level=inf_ob.getMaxLevel()
    if level>max_level:
        temp = np.ones((480, 640,3), dtype=np.uint8)
        temp*=255
        cv2.putText(temp,"You have completed all levels of this game",(50,220),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0,),2,cv2.LINE_AA)  
        cv2.imshow('Message',temp)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        return

    
    """cnfg=pd.read_csv('configure.csv')
    cnfg=cnfg.iloc[:,:].values
    game_no=0
    while game_no<int(cnfg[-1][1]):
        if str(cnfg[game_no][1]).lower()=='gameofcolors': break
        game_no=game_no+1
    """
    
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
        
    
    colors=("Red","Blue","Green")
    
    ranges={"Red":[[170,100,0],[180,255,255]],"Blue":[[110,50,50],[130,255,255]],"Green":[[36,25,25],[70,255,255]]}
    
    min_area=1200
    max_area=10000
    
            
    kernel = np.ones((5, 5), np.uint8)
            
    camera = cv2.VideoCapture(0)
    #max_level=int(cnfg[game_no][2])
    
    
    def isLevelUp(scores):
        total=0
        for score in scores:
            total=total+score
            
        avg=total/(len(scores)*100)
        if avg>=0.75: 
            """uid=int(cnfg[0][0])
            sum_l=0
            for i in range(game_no):
                sum_l=sum_l+cnfg[i][2]
                
            idx=uid*int(cnfg[-1][2])+sum_l+(level-1)
            d_scores=pd.read_csv('scores.csv')
            os.remove('scores.csv')
            d_scores.iloc[idx][0]=avg*100
            d_scores.to_csv('scores.csv',index=False)"""
            inf_ob.update_scores(level,avg)
            return True
        else: return False
    
    def check(img,image,cnt):
        n_image=image.copy()
        n_img=img.copy()
        n_image = cv2.cvtColor(n_image, cv2.COLOR_BGR2GRAY)
        n_img = cv2.cvtColor(n_img, cv2.COLOR_BGR2GRAY)
        x, y, w, h = cv2.boundingRect(cnt)
        img1 =n_img[y+5:y + h - 5, x+5:x + w - 5]
        img2 =n_image[y+5:y + h - 5, x+5:x + w - 5]
        img1=np.asarray(img1)
        img2=np.asarray(img2)
        return np.array_equal(img1,img2)
    
    def preprocess(image,mapping):
        for color in colors:
            lower_range=np.asarray(ranges[color][0])
            upper_range=np.asarray(ranges[color][1])
            hsv_im=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            mask_im=cv2.inRange(hsv_im,lower_range,upper_range)
            ret, thresh = cv2.threshold(mask_im, 127, 255, 0)
            thresh=cv2.dilate(thresh,kernel,iterations=2)
            contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            mapping[color]=contours    
    
    def getContour(cnts):
        cnt=None
        cur_area=0
        for ct in cnts:
            area=float(cv2.contourArea(ct))
            if area>=min_area and area<=max_area and area>cur_area:
                cur_area=area
                cnt=ct
        return cnt
    
    
    #level=1
    br=True
    
    while level<=max_level:
        temp = np.ones((480, 640,3), dtype=np.uint8)
        temp*=255
        cv2.putText(temp,"Level"+str(level),(200,220),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0,),2,cv2.LINE_AA)
        temp_2=cv2.resize(temp,(1400,750))
        cv2.imshow('Frame',temp_2)
        cv2.waitKey(5000)
        img = cv2.imread('images/ColorGame/ColorGameImg'+str(level)+'.jpg')
        img = cv2.resize(img, (640,480), interpolation = cv2.INTER_AREA)
        mapping={}
        preprocess(img,mapping)
        scores=[]    
        dist_colors=[]
        for color in colors:
            if len(mapping[color])==0: continue
            dist_colors.append(color)
        random.shuffle(dist_colors)
        
        for d_color in dist_colors:
            temp = np.ones((480, 640,3), dtype=np.uint8)
            temp*=255
            cv2.putText(temp,"Pick all '"+d_color+"'",(200,220),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)
            temp_2=cv2.resize(temp,(1400,750))
            cv2.imshow('Frame',temp_2)
            cv2.waitKey(5000)
            count=len(mapping[d_color])
            cur_score=0
            tcount=0
            image=img.copy() 
            cv2.putText(image,"Pick all '"+d_color+"'",(270,460),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0,),1,cv2.LINE_AA)
            while count>0:
        # Grab the frame
                    (grabbed, frame) = camera.read()
                    if not grabbed:
                        break
                    frame = cv2.flip(frame, 1)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    
                    # Check to see if we have reached the end of the video 
                    # (useful when input is a video file not a live video stream)
                    if not grabbed:
                        break
                    
                    
                    
                    
                # Determine which pixels fall within the blue boundaries and then blur the binary image
                    blueMask1 = cv2.inRange(hsv, blueLower1, blueUpper1)
                    blueMask2 = cv2.inRange(hsv, blueLower2, blueUpper2)
                    blueMask=blueMask1|blueMask2
                    blueMask = cv2.erode(blueMask, kernel, iterations=2)
                    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
                    blueMask = cv2.dilate(blueMask, kernel, iterations=1)
                    # Find contours (bottle cap in my case) in the image
                    (cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    #center=None
                    
                    
                    
                    
                # Check to see if any contours were found
                    	# Sort the contours and find the largest one -- we
                    	# will assume this contour correspondes to the area of the bottle cap
                    if len(cnts)>0:    
                        cnt=getContour(cnts)
                        center=None
                        if cnt is not None:
                            # Get the radius of the enclosing circle around the found contour
                            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                            center=(int(x),int(y))
                                # Draw the circle around the contour
                            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                                # Get the moments to calculate the center of the contour (in this case Circle)
                            
                            for color in mapping.keys():
                                flag=True
                                for fcnt in mapping[color]:
                                    dist= cv2.pointPolygonTest(fcnt,center,True)
                                    if dist>=0:
                                        M = cv2.moments(fcnt)
                                        centerX =int(M['m10'] / M['m00'])
                                        centerY =int(M['m01'] / M['m00'])
                                        if color==d_color:
                                            if check(img,image,fcnt): 
                                                count=count-1
                                                tcount=tcount+1
                                                cur_score=cur_score+20
                                            image = cv2.line(image, (centerX-5,centerY-10) , (centerX+5,centerY+10), (0,0,0), 2)
                                            image=cv2.line(image,(centerX+5,centerY+10),(centerX+25,centerY-25),(0,0,0), 2)  
                                            
                                        else:
                                            if check(img,image,fcnt): 
                                                tcount=tcount+1
                                                cur_score=cur_score-5
                                            image = cv2.line(image, (centerX,centerY) , (centerX+20,centerY+20), (0,0,0), 2)
                                            image=cv2.line(image,(centerX,centerY+20),(centerX+20,centerY),(0,0,0), 2)  
                                        flag=False
                                        break
                                if flag==False: break
                            
                            
                            image_up=image.copy()    
                            if center is not None: cv2.circle(image_up,center,5,(0,255,255),-1)
                            image_up_2=cv2.resize(image_up,(1400,750))
                            cv2.imshow("Frame",image_up_2)
                            if count==0: cv2.waitKey(1000)
                            
                    else: 
                        image_2=cv2.resize(image,(1400,750))
                        cv2.imshow("Frame",image_2)
                    
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
