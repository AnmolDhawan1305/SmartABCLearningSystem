# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:25:15 2019

@author: HP 250 G5
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:56:42 2019

@author: HP 250 G5
"""


import cv2
import numpy as np
from playsound import playsound

words={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'teen':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,'hundred':'hundred','thousand':'thousand'}

def Helper(n):
    if n[0]=='0':
        if len(n)==1: return ''
        else: return Helper(n[1:])
    if len(n)<=2:
        if n=='1': return 'one'
        elif n=='2': return 'two'
        elif n=='3': return 'three'
        elif n=='4': return 'four'
        elif n=='5': return 'five'
        elif n=='6': return 'six'
        elif n=='7': return 'seven'
        elif n=='8': return 'eight'
        elif n=='9': return 'nine'
        elif n=='10': return 'ten'
        elif n=='11': return 'eleven'
        elif n=='12': return 'twelve'
        elif n=='13': return 'thirteen'
        elif n=='14': return 'fourteen'
        elif n=='15': return 'fifteen'
        elif n=='16': return 'sixteen'
        elif n=='17': return 'seventeen'
        elif n=='18': return 'eighteen'
        elif n=='19': return 'nineteen'
        elif n=='20': return 'twenty'
        elif n[0]=='2': return ('twenty '+Helper(n[1]))
        elif n[0]=='3': return ('thirty '+Helper(n[1]))
        elif n[0]=='4': return ('forty '+Helper(n[1]))
        elif n[0]=='5': return ('fifty '+Helper(n[1]))
        elif n[0]=='6': return ('sixty '+Helper(n[1]))
        elif n[0]=='7': return ('seventy '+Helper(n[1]))
        elif n[0]=='8': return ('eighty '+Helper(n[1]))
        elif n[0]=='9': return ('ninety '+Helper(n[1]))
        
    elif len(n)==3:
        temp=Helper(n[0])
        temp+=' hundred'
        if n[1]!='0' or n[2]!='0': temp+=' '
        temp+=Helper(n[1:])
        return temp
        
    elif len(n)==4:
        return (Helper(n[0])+' thousand '+Helper(n[1:]))
    
    elif len(n)==5:
        return (Helper(n[0:2])+' thousand '+Helper(n[2:]))


def convert(n):
    s=Helper(n)
    frame=np.ones((150,700,3),dtype=np.uint8)
    frame*=255
    textsize = cv2.getTextSize(s,cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    textX = (frame.shape[1] - textsize[0]) / 2
    textY = (frame.shape[0] + textsize[1]) / 2
    tempX=textX
    
    sent=s.split()
    
    for i in range(0,len(sent)):
        textX=tempX
        if i>=1:
            s1=' '.join(sent[:i])
            cv2.putText(frame,s1,(int(textX),int(textY)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
            textsize = cv2.getTextSize(s1,cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
            textX=textX+textsize
        
        cv2.putText(frame,' '+sent[i]+' ',(int(textX),int(textY)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        textsize = cv2.getTextSize(' '+sent[i]+' ',cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
        textX=textX+textsize
        
        if (i+1)<len(sent):
            s2=' '.join(sent[(i+1):])
            cv2.putText(frame,s2,(int(textX),int(textY)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
        
        cv2.imshow('Result',frame)
        cv2.waitKey(1)
        filename = 'audios/'+str(words[sent[i]])+'.wav'
        playsound(filename)
        frame=np.ones((150,700,3),dtype=np.uint8)
        frame*=255    
    
    cv2.putText(frame,s,(int(tempX),int(textY)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
    cv2.imshow('Result',frame)
    frame=np.ones((150,700,3),dtype=np.uint8)
    frame*=255
    cv2.waitKey(0)
    cv2.destroyWindow('Result')