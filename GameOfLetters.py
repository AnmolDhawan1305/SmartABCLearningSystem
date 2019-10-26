# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:26:20 2019

@author: HP 250 G5
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:13:17 2019

@author: HP 250 G5
"""

from keras.models import load_model
import numpy as np
import cv2

# Load the models built in the previous steps
cnn_model = load_model('emnist_cnn_model.h5')

# Letters lookup
letters = { 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T',
21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: 'NOT'}

image = cv2.imread('images/LetterGameImg.jpg')
image = cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray=cv2.bitwise_not(gray)
#cv2.imshow("Original Image",image)
#ret, thresh = cv2.threshold(gray, 127, 255, 0)\
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.imshow("Working Image",thresh)
_,contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
#print(len(contours))

mapping={}
for l in letters.values():
    mapping[l]=[]
    
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    alphabet = gray[y+5:y + h - 5, x+5:x + w - 5]
    #cv2.imshow("Cropped Letter",alphabet)
    #cv2.waitKey(0)
    newImage = cv2.resize(alphabet, (28, 28),interpolation = cv2.INTER_AREA)
    newImage = np.array(newImage)
    newImage = newImage.astype('float32')/255
    prediction = cnn_model.predict(newImage.reshape(1,28,28,1))[0]
    prediction = np.argmax(prediction)
    #cv2.putText(image,str(letters[int(prediction)+1]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)    
    predicted_letter=str(letters[int(prediction)+1])
    mapping[predicted_letter].append(cnt)

d_letter='A'

blueLower = np.array([110,50,50])
blueUpper = np.array([130,255,255])

kernel = np.ones((5, 5), np.uint8)

camera = cv2.VideoCapture(0)

#dist = cv2.pointPolygonTest(cnt,(50,50),True)

# Keep looping
while True:
    # Grab the frame
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
            
            for letter in mapping.keys():
                flag=True
                for cnt in mapping[letter]:
                    dist= cv2.pointPolygonTest(cnt,center,True)
                    if dist>=0:
                        if letter==d_letter:
                            cv2.drawContours(image, [cnt], 0, (0, 255, 0), -1)

                        else: 
                            cv2.drawContours(image, [cnt], 0, (0, 0, 255), -1)
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
