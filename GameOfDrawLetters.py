# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:35:39 2019

@author: HP 250 G5
"""

from keras.models import load_model
from collections import deque
import numpy as np
import simpleaudio as sa
import cv2
import random
from playsound import playsound
from interface import Interface

def game_of_draw_letters(color_point,color_pen,level):
    level=int(level)
    inf_ob=Interface('gameofdrawletters')
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

    # Load the models built in the previous steps
    cnn_model = load_model('emnist_cnn_model.h5')
    
    # Letters lookup
    letters = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
    11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
    21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}
    
    # Define the upper and lower boundaries for a color to be considered "Blue"
    min_area=1200
    max_area=10000
    
    
    
    # Define a 5x5 kernel for erosion and dilation
    kernel = np.ones((5, 5), np.uint8)
    
    # Define Black Board
    blackboard = np.zeros((480,640,3), dtype=np.uint8)
    alphabet = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Setup deques to store alphabet drawn on screen
    points = deque(maxlen=512)
    
    # Define prediction variables
    
    prediction = 26
    
    
    
    def getContour(cnts):
        cnt=None
        cur_area=0
        for ct in cnts:
            area=float(cv2.contourArea(ct))
            if area>=min_area and area<=max_area and area>cur_area:
                cur_area=area
                cnt=ct
        return cnt
    
    
    camera = cv2.VideoCapture(0)
    br=True
    while level<=max_level and br:
        cv2.namedWindow('Draw this!', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Draw this!', 640, 480)
        temp = np.ones((480, 640,3), dtype=np.uint8)
        temp*=255
        cv2.putText(temp,"Level"+str(level),(200,220),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0,),2,cv2.LINE_AA)
        cv2.imshow('Draw this!',temp)
        cv2.waitKey(3000)
        
        d_char=letters[random.randrange(1,27,1)]
        
        d=cv2.imread('images/DrawGame/draw_this/draw_'+d_char+'.PNG')
        prob_dist=[]
        trials=0
        thresh_t=5
        crs=0
        count=0
        while True:
            # Grab the current paintWindow
            try:
                if level==1: cv2.imshow('Draw this!',d)
                elif level==2:
                    t = np.ones((480, 640,3), dtype=np.uint8)
                    t*=255
                    cv2.putText(t,"Listen to the sound and draw letter.",(70,220),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0,),2,cv2.LINE_AA)
                    cv2.imshow('Draw this!',t)
                    if count==0:
                        cv2.waitKey(1000)
                        filename = 'audios/'+d_char.upper()+'.wav'
                        playsound(filename)
                        count=500
                
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
        
                
                cnt=getContour(cnts_blue)
                if cnt is not None: 
                    ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                    # Draw the circle around the contour
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    M = cv2.moments(cnt)
                    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                    points.appendleft(center)
            
                else:
                    if len(points) != 0:
                        blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                        blur1 = cv2.medianBlur(blackboard_gray, 15)
                        blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                        thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                        blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
                        if len(blackboard_cnts) >= 1:
                            cnt = sorted(blackboard_cnts, key = cv2.contourArea, reverse = True)[0]
            
                            if cv2.contourArea(cnt) > 1000:
                                x, y, w, h = cv2.boundingRect(cnt)
                                alphabet = blackboard_gray[y-10:y + h + 10, x-10:x + w + 10]
                                newImage = cv2.resize(alphabet, (28, 28))
                                newImage = np.array(newImage)
                                newImage = newImage.astype('float32')/255
            
                                
                                
                                prediction = cnn_model.predict(newImage.reshape(1,28,28,1))[0]
                                prob_dist=prediction
                                prediction = np.argmax(prediction)
                                #trials=trials-1
            
                        # Empty the points deque and the blackboard
                        points = deque(maxlen=512)
                        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
                        
                    if len(cnts_red) > 0:
                             cnt=getContour(cnts_red)
                             if cnt is not None:
                                # Get the radius of the enclosing circle around the found contour
                                 ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                                    # Draw the circle around the contour
                                 center =(int(x),int(y))
                                 cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                                 cv2.circle(blackboard, center,5,(0,255,0),-1)
            
                # Connect the points with a line
                
                for i in range(1, len(points)):
                        if points[i - 1] is None or points[i] is None:
                                continue
                        #flag=0
                        #cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 2)
                        cv2.line(blackboard, points[i - 1], points[i], (255, 255, 255), 8)
            
                
                    
                # Put the result on the screen
                #cv2.putText(frame, "Multilayer Perceptron : " + str(letters[int(prediction1)+1]), (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 2)
               # cv2.putText(frame, "Convolution Neural Network:  " + str(letters[int(prediction)+1]), (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                predicted_char=str(letters[int(prediction)+1])
                if predicted_char==d_char:
                    rating=''
                    if prob_dist[prediction]>=0.95: rating='Bravo'
                    elif prob_dist[prediction]>=0.85: rating='Excellent' 
                    elif prob_dist[prediction]>=0.75: rating='Very_good'
                    elif prob_dist[prediction]>=0.50: rating='Good'
                    else: rating='Fair'
                    if prob_dist[prediction]>=0.85: crs=crs+1
                    filename = 'audios/'+rating+'.wav'
                    wave_obj = sa.WaveObject.from_wave_file(filename)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()  
                    cv2.waitKey(33)
                    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Result', 640, 480)
                    x=cv2.imread('images/DrawGame/objects/object_'+d_char+'.jpg')
                    cv2.imshow('Result',x)
                    cv2.waitKey(35)
                    filename = 'audios/audio_'+d_char+'.wav'
                    wave_obj = sa.WaveObject.from_wave_file(filename)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()  # Wait until sound has finished playing
                    cv2.waitKey(0)
                    prediction=26
                    cv2.destroyWindow('Result')
                    cv2.waitKey(33)
                    d_char=letters[random.randrange(1,27,1)]
                    d=cv2.imread('images/DrawGame/draw_this/draw_'+d_char+'.PNG')
                    count=0
                    thresh_t=thresh_t-1
                    trials=trials+1
                 
                elif predicted_char!='-':
                    filename = 'audios/try_again.wav'
                    wave_obj = sa.WaveObject.from_wave_file(filename)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()  # Wait until sound has finished playing
                    prediction=26
                    trials=trials+1
                    points = deque(maxlen=512)
                    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
                    count=0
                    
                    
                    
                # Show the frame
                cv2.imshow("Board", blackboard)
                
                
                if len(points)==0:
                    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
                
                if thresh_t==0:
                    #trials=5
                    score=float(crs)/trials
                    if score<0: score=0
                    if score>=0.7:
                        inf_ob.update_scores(level,score)
                        temp = np.ones((480, 640,3), dtype=np.uint8)
                        temp*=255
                        if level<max_level: cv2.putText(temp,"Congrats, Level UP!!",(200,220),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)
                        else: cv2.putText(temp,"Congrats, all levels completed",(90,220),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)  
                        cv2.imshow('Draw this!',temp)
                        cv2.waitKey(5000)
                        level=level+1
                        break
                    else:
                        temp = np.ones((480, 640,3), dtype=np.uint8)
                        temp*=255
                        cv2.putText(temp,"Try again Level "+str(level)+"!!",(200,220),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,),2,cv2.LINE_AA)
                        cv2.imshow('Draw this!',temp)
                        cv2.waitKey(5000)
                        break
                if(level==2 and count>0): count=count-1     
            except:
                print('exception occurs')
                pass
            
            # If the 'q' key is pressed, stop the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                br=False
                break
        
    # Cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()