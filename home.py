# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:30:13 2019

@author: hp
"""

#double coolors  game of draw digits   game of draw letters   esign   gameofnum2words
from tkinter import *
#from PIL import ImageTk,Image

import tkinter as tk
import cv2
from tkinter.messagebox import *
from GameOfDigits import game_of_digits
from GameOfDrawDigits import game_of_draw_digits
from GameOfDrawLetters import game_of_draw_letters
from GameOfLetters import game_of_letters
from GameOfColors import game_of_colors
from GameOfShapes import game_of_shapes
from GameOfNum2Words import game_of_num2words
from eSign import eSign
from interface import Interface


color_point  = 'Red'
color_pen = 'Blue'
video_name='gameofdrawletters'
def start():
    
    #color_pen
    #color_point
    
    """def game_of_digits(a, b):
        "a"
       
    def game_of_draw_digits(a, b):
        "a"
       
    def game_of_draw_letters(a, b):
        "a"
       
    def game_of_letters(a, b):
        "a"
       
    def game_of_colors(a, b):
        "a"
       
    def game_of_shapes(a, b):
        "a"
       
    def game_of_num2words(a, b):
        "a"
       
    def esign(a):
        "a"
    """
    
    def play_demo():
        path='videos/'+video_name+'.mp4'
        cap = cv2.VideoCapture(path) 
        cv2.namedWindow('Demonstration Video', cv2.WINDOW_NORMAL)
        # Check if camera opened successfully 
        if (cap.isOpened()== False):  
          print("Error opening video  file") 
           
        # Read until video is completed 
        while(cap.isOpened()): 
              
          # Capture frame-by-frame 
          ret, frame = cap.read() 
          if ret == True: 
           
            # Display the resulting frame 
            cv2.resize(frame,(1400,750))
            cv2.imshow('Demonstration Video', frame) 
           
            # Press Q on keyboard to  exit 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
              break
           
          # Break the loop 
          else:  
            break
           
        # When everything done, release  
        # the video capture object 
        cap.release() 
           
        # Closes all the frames 
        cv2.destroyAllWindows()

    
    
    l3text = """Game Of Draw Letters\n
    Introduction:
    In this app user will learn how to draw letter correctly.
    
    It has 2 levels:
    
    Level 1:
    In this level user will get the hint of letter in the form of dotted letter which later on he/she will draw precisely. If user will draw correct letter the image of \n letter and its audio will pop on screen.
    
    Level 2:
    In this level audio will play and user has to detect the audio of letter and draw it, If its wrong the message will show on screen to draw again, if its \n correct  user will be evaluated accordingly and its image and audio will show on screen.
    
    Scoring criteria:
    A response is considered legit if the accuracy of drawn character is greater than or equal to 85%.
    Score=Number of legit responses/Number of Trials.
    Here, Number of trials is set to 5(default).
    
    Criteria for level up:
    If Score>=80%, then level up, otherwise not.
    
    Skills assessed:
    With this app user will develop the skill of writting the letter correctly because of hint and will also learn how to spell it correctly.
     """
    l4text = """Game Of Draw Digits\n
    Introduction:
    In this app user will learn how to draw digit correctly.
    
    It has 2 levels:
    
    Level 1:
    In this level user will get the hint of digit in the form of dotted digit which later on he/she will draw precisely. If user will draw correct digit the image of
    \n digit and its audio will pop on screen.
    
    Level 2:
    In this level audio will play and user has to detect the audio of digit and draw it, If its wrong the message will show on screen to draw again, if its correct \n user will be evaluated accordingly and its image and audio will show on screen.
    
    Scoring criteria:
    A response is considered legit if the accuracy of drawn character is greater than or equal to 85%.
    Score=Number of legit responses/Number of Trials.
    Here, Number of trials is set to 5(default).
    
    Criteria for level up:
    If Score>=80%, then level up, otherwise not.
    
    Skills assessed:
    With this app user will develop the skill of writting the digit correctly because of hint and will also learn how to spell it correctly. """
    
    
    l5text = """Game of Digits \n
    Introduction:
    In this app user has to choose the digit which is given in question.
    
    It has 2 levels:
    Level 1:
    In this Level total 9 digits are given to user from which user will select the given digit, If he/she pick the wrong digit the box of digit will convert in red \n color and user has to choose again otherwise if user will pick right digit it will be of green color.
    
    Level 2:
    In this Level user has total 16 number of digits, from them user will detect correct digit, checking criteria will be same as level 0.
    
    Scoring criteria:
    +20 points for each correct response.
    -5 points for each wrong response.
    In each round, score=Total points gained/Max points.
                   
    Criteria for level up:
    Let Avg be average score in a level.
    Avg=sum of scores of all round/No of rounds.
    If Avg>=75%, then level up, otherwise not.
    
    Skills assessed:
    With this app user will develop the skill of detecting the digit fastly.
     """
     
    l6text = """Game Of Letter \n
    Introduction:
    In this app user has to choose the letter which is given in question.
    
    It has 2 levels:
    Level 1:
    In this Level total 9 letters are given to user from which user will select the given letter, If he/she pick the wrong letter the box of letter will convert in red \n color and user has to choose again otherwise if user will pick right letter it will be of green color.
    
    Level 2:
    In this Level user has total 16 number of letters, from them user will detect correct letter, checking criteria will be same as level 0.
    
    Scoring criteria:
    +20 points for each correct response.
    -5 points for each wrong response.
    In each round, score=Total points gained/Max points.
                   
    Criteria for level up:
    Let Avg be average score in a level.
    Avg=sum of scores of all round/No of rounds.
    If Avg>=75%, then level up, otherwise not.
    
    Skills assessed:
    With this app user will develop the skill of detecting the letter fastly.
     """
     
    l7text = """ Game Of Color \n
    Introduction:
    In this app various colors will be available to user and he/she have to pick color which will be mentioned in question.
    
    It has 2 levels:
    Level 1:
    In this Level random colors will be given to user from which user will select the given color, If he/she pick the wrong color the cross sign will be shown on \n the box of color and user has to choose again otherwise if user will pick right color the tick sign will be shown on it.
    
    Level 2:
    In this Level more number of colors will be given and the user have to select the right one, checking criteria will be same as level 0.
    
    Scoring criteria:
    +20 points for each correct response.
    -5 points for each wrong response.
    In each round, score=Total points gained/Max points.
                   
    Criteria for level up:
    Let Avg be average score in a level.
    Avg=sum of scores of all round/No of rounds.
    If Avg>=75%, then level up, otherwise not.
    
    Skills assessed:
    With this app user will develop the skill of detecting the color.
    
    """
    l8text = """Game Of Shape \n
    Introduction:
    In this app user have to pick the right shape from other random shapes which is given in question.
    
    It has 2 levels:
    Level 1:
    In this Level there are various shapes given to him from them user will select the given shape, If he/she pick the wrong shape the box of shape will \n convert in red color and user has to choose again otherwise if user will pick right shape it will be of green color.
    
    Level 2:
    In this Level there will be increse in shapes as compared to previous level and user have to detect correct shape, checking criteria will be same as level 0.
    
    Scoring criteria:
    +20 points for each correct response.
    -5 points for each wrong response.
    In each round, score=Total points gained/Max points.
                   
    Criteria for level up:
    Let Avg be average score in a level.
    Avg=sum of scores of all round/No of rounds.
    If Avg>=75%, then level up, otherwise not.
    
    Skills assessed:
    With this app user will develop the skill of detecting the shapes.
     """
     
     
    l9text = """ Game Of Num to Words \n
    
    Introduction:
    
    In this app user will write number on screen and will get the result in form of words.
    After writting a number a timer of 8 seconds will shown on screen which is guessing time for our number. After 8 seconds the number in form of words \n will shown and the user will also get audio of that number.
    
    
    """
    l10text = """ E - Sign buttons \n
    
    Introduction:
    Its a feature which can record any signature of individual using finger gesture and you can save it as well.
    """
    
    l11text = """ Instructions
    
    Follow below points for smooth experience-\n
    1. Always use it in moderate light conditions and do not use it in direct sunlight. \n
    2. Do not use it outdoors.\n
    3. Only use object which is provided.\n
    4. Do not use object with colour which resembles any background colour.\n
    5. Keep webcam parallel to object, or laptop lid at right angle or an acute angle.\n
    6. Keep object between 20cm and 50cm of distance from webcam. \n
    """
    window = tk.Tk()
    
    window.title('games of fun')
    window.geometry('1400x750')
    window.configure(background='pale turquoise') 
    #filename=tk.PhotoImage('C:\Users\hp\Desktop\modifications capstone\Images\yesornot.gif')
    #img = ImageTk.PhotoImage(Image.open("images/bg2.jpg"))
    
    #window= tk.Label(window_, image=  filename)
    #window.place(x=0, y=0)
    #window.pack()
    l1 = tk.Label(window, text = "Game World...Its Time To Play",font=("Arial Bold",60), bg= "pale turquoise")
    
    
    l1.place(x=700,y=2, anchor = N)
    
    l2 = tk.Label(window, text = "Games", font = ("Arial",40), bg= "pale turquoise")
    l2.place(x=590,y=90)
    
    
    
    l3 = tk.Label(window, text = "1. Game Of Draw Letters",font=("Arial ",20), bg= "pale turquoise")
    l3.place(x=300,y=180)
    
    def gameOfDrawLetterspl():
        win3_ = tk.Tk()
        win3_.title("Begin Game of Draw Letters")
        win3_.geometry('600x300')
        l3play1 = tk.Label(win3_ , text = "Select the color for pen", font = ("Arial",15),justify = CENTER)
        l3play1.place(x=15,y=15)
       
       
        l3play2 = tk.Label(win3_ , text = "Select the color for pointer", font = ("Arial",15),justify = CENTER)
        l3play2.place(x=15,y=40)
        
        l3play3 = tk.Label(win3_ , text = "Select the level you want to play", font = ("Arial",15),justify = CENTER)
        l3play3.place(x=15,y=65)
        l3playvar3 = tk.StringVar(win3_)
        
        inf_ob=Interface('gameofdrawletters')
        l_scores=inf_ob.completed_levels()
        levels =[]
        for i in range(len(l_scores)+1,0,-1):
            levels.append(str(i))
        
        
        l3playvar3.set(levels[0])
        popupMenu3 = tk.OptionMenu(win3_, l3playvar3, *levels)
        popupMenu3.place(x=475,y=65)
            
        
        l3playvar1 = tk.StringVar(win3_)
        l3playvar2 = tk.StringVar(win3_)
        
       
        choices = { 'Red','Blue','Green'}
        l3playvar1.set('Blue')
        l3playvar2.set('Red')
        
        # set the default option
    
        popupMenu1 = tk.OptionMenu(win3_, l3playvar1, *choices)
        popupMenu1.place(x=475,y=15)
       
        popupMenu2 = tk.OptionMenu(win3_, l3playvar2, *choices)
        popupMenu2.place(x=475,y=40)
        
    
       
        l3note = tk.Label(win3_, text = "Press 'q' to exit",font = ("Arial",15),justify = CENTER)
        l3note.place(x=220,y=100)   
    
        name=inf_ob.getName()
        
        label = tk.Label(win3_ , text = "Levels completed by "+name+"-", font = ("Arial",15),justify = CENTER)
        label.place(x=120,y=170)     
       
        def launch_game_of_draw_letters():
            global color_point,color_pen
            if str(l3playvar1.get())==str(l3playvar2.get()):
                showerror("Error", "color for pointer and color for pen can't be same.")
                return
            color_pen = str(l3playvar1.get())
            color_point = str(l3playvar2.get())
            level=str(l3playvar3.get())
            game_of_draw_letters(color_point,color_pen,level)
           
           
        l3play = tk.Button(win3_, text = "Ok",command = launch_game_of_draw_letters)
        l3play.pack()
        l3play.place(x = 280, y = 138)
        
            
        root=tk.Frame (win3_, width=200, height=200)
        root.place(x=225,y=200)
    
        for i in range(len(l_scores)-1,-1,-1):
            b1=tk.Label(root, text="Level "+str(i+1)+":", font = ("Arial",13) ) 
            b1.grid(row=len(l_scores)-i-1, column=0)
            b1 = tk.Label(root, text=str(l_scores[i])+"%", font = ("Arial",13) )
            b1.grid(row=len(l_scores)-i-1, column=1)
       
    
    def gameOfDrawLetters():
        global video_name
        win3 = tk.Tk()
        win3.title("Game Of Draw Letters")
        win3.geometry('1400x750')
        l3_ = tk.Label(win3, text = l3text, font = ("Arial",15),justify = LEFT)
        l3_.place(x=15,y=50)
        video_name='gameofdrawletters'
        b3_ = tk.Button(win3, text = "Watch Demonstration Video", command = play_demo )
        b3_.pack()
        b3_.place(x=550, y=650)
        b3_.config(width = 40,bg = 'green',fg = 'white')
    
       
    
       
    
    b3 = tk.Button(window, text = "Know more",command = gameOfDrawLetters)
    b3.pack()
    b3.place(x=850,y=180)
    b3.config(width = 15,bg = 'light blue2')
    
    b3_ = tk.Button(window, text = "Play",command = gameOfDrawLetterspl)
    b3_.pack()
    b3_.place(x=1000,y=180)
    b3_.config(width = 15,bg = 'lime green',fg = 'black')
    
    
    l4 = tk.Label(window, text = "2. Game Of Draw Digits",font=("Arial ",20), bg= "pale turquoise")
    l4.place(x=300,y=220)
    
    def gameOfDrawDigitspl():
        win4_ = tk.Tk()
        win4_.title("Begin Game of Draw Digits")
        win4_.geometry('600x300')
        l4play1 = tk.Label(win4_ , text = "Select the color for pen", font = ("Arial",15),justify = CENTER)
        l4play1.place(x=15,y=15)
       
       
        l4play2 = tk.Label(win4_ , text = "Select the color for pointer", font = ("Arial",15),justify = CENTER)
        l4play2.place(x=15,y=40)
        
        l4play4 = tk.Label(win4_ , text = "Select the level you want to play", font = ("Arial",15),justify = CENTER)
        l4play4.place(x=15,y=65)
        l4playvar4 = tk.StringVar(win4_)
        
        
        inf_ob=Interface('gameofdrawdigits')
        l_scores=inf_ob.completed_levels()
        levels =[]
        for i in range(len(l_scores)+1,0,-1):
            levels.append(str(i))
        
        
        l4playvar4.set(levels[0])
        popupMenu3 = tk.OptionMenu(win4_, l4playvar4, *levels)
        popupMenu3.place(x=475,y=65)
       
        l4playvar1 = tk.StringVar(win4_)
        l4playvar2 = tk.StringVar(win4_)
       
        choices = { 'Red','Blue','Green'}
        l4playvar1.set('Blue')
        l4playvar2.set('Red')
        # set the default option
    
        popupMenu1 = tk.OptionMenu(win4_, l4playvar1, *choices)
        popupMenu1.place(x=475,y=15)
       
        popupMenu2 = tk.OptionMenu(win4_, l4playvar2, *choices)
        popupMenu2.place(x=475,y=40)
       
        l4note = tk.Label(win4_, text = "Press 'q' to exit",font = ("Arial",15),justify = CENTER)
        l4note.place(x=220,y=100)
        
        name=inf_ob.getName()
        
        label = tk.Label(win4_ , text = "Levels completed by "+name+"-", font = ("Arial",15),justify = CENTER)
        label.place(x=120,y=170)     
       
        def launch_game_of_draw_digits():
            global color_point,color_pen
            if str(l4playvar1.get())==str(l4playvar2.get()):
                showerror("Error", "color for pointer and color for pen can't be same.")
                return
            color_pen = str(l4playvar1.get())
            color_point = str(l4playvar2.get())
            level=str(l4playvar4.get())

            game_of_draw_digits(color_point,color_pen,level)
           
           
        l4play = tk.Button(win4_, text = "Ok",command = launch_game_of_draw_digits)
        l4play.pack()
        l4play.place(x = 280, y = 138)
        
        root=tk.Frame (win4_, width=200, height=200)
        root.place(x=225,y=200)
        
        
        for i in range(len(l_scores)-1,-1,-1):
            b1=tk.Label(root, text="Level "+str(i+1)+":", font = ("Arial",13) ) 
            b1.grid(row=len(l_scores)-i-1, column=0)
            b1 = tk.Label(root, text=str(l_scores[i])+"%", font = ("Arial",13) )
            b1.grid(row=len(l_scores)-i-1, column=1)
       
    
    def gameOfDrawDigits():
        global video_name
        win4 = tk.Tk()
        win4.title("Game Of Draw Digits")
        win4.geometry('1400x750')
        l4_ = tk.Label(win4, text = l4text, font = ("Arial",15),justify = LEFT)
        l4_.place(x=15,y=50)
        video_name='gameofdrawdigits'
        b4_ = tk.Button(win4, text = "Watch Demonstration Video", command = play_demo )
        b4_.pack()
        b4_.place(x=550, y=660)
        b4_.config(width = 40,bg = 'green',fg = 'white')
       
    
    b4 = tk.Button(window, text = "Know more",command = gameOfDrawDigits)
    b4.pack()
    b4.place(x=850,y=220)
    b4.config(width = 15,bg = 'light blue2')
    
    b4_ = tk.Button(window, text = "Play", command = gameOfDrawDigitspl)
    b4_.pack()
    b4_.place(x=1000,y=220)
    b4_.config(width = 15,bg = 'lime green',fg = 'black')
    
    
    l5 = tk.Label(window, text = "3. Game of Digits",font=("Arial ",20), bg= "pale turquoise")
    l5.place(x=300,y=260)
    
    def gameOfDigits():
        win5 = tk.Tk()
        global video_name
        win5.title("Game Of Digits")
        win5.geometry('1400x750')
        l5_ = tk.Label(win5, text = l5text, font = ("Arial",15),justify = LEFT)
        l5_.place(x=15,y=50)
        video_name='gameofdigits'
        b5_ = tk.Button(win5, text = "Watch Demonstration Video", command = play_demo )
        b5_.pack()
        b5_.place(x=550, y=650)
        b5_.config(width = 40,bg = 'green',fg = 'white')
       
       
    def gameOfDigitspl():
        win5_ = tk.Tk()
        win5_.title("Begin Game Of Digits")
        win5_.geometry('600x300')  
        l5play = tk.Label(win5_ , text = "Select the color for pointer", font = ("Arial",15),justify = CENTER)
        l5play.place(x=15,y=15)
        #print(l5play.winfo_width())
        l5playvar = tk.StringVar(win5_)
       
        choices = { 'Red','Blue','Green'}
        l5playvar.set('Blue') # set the default option
    
        popupMenu = tk.OptionMenu(win5_, l5playvar, *choices)
        popupMenu.place(x=475,y=15)
        
        l5play3 = tk.Label(win5_ , text = "Select the level you want to play", font = ("Arial",15),justify = CENTER)
        l5play3.place(x=15,y=40)
        l5playvar3 = tk.StringVar(win5_)
        
        inf_ob=Interface('gameofdigits')
        l_scores=inf_ob.completed_levels()
        levels =[]
        for i in range(len(l_scores)+1,0,-1):
            levels.append(str(i))
        
        
        l5playvar3.set(levels[0])
        popupMenu3 = tk.OptionMenu(win5_, l5playvar3, *levels)
        popupMenu3.place(x=475,y=40)
       
        l5note = tk.Label(win5_, text = "Press 'q' to exit",font = ("Arial",15),justify = CENTER)
        l5note.place(x=220,y=100)
        
        name=inf_ob.getName()
        
        label = tk.Label(win5_ , text = "Levels completed by "+name+"-", font = ("Arial",15),justify = CENTER)
        label.place(x=120,y=170)     
       
        def launch_game_of_digits():
            global color_point
            color_point = str(l5playvar.get())
            level=str(l5playvar3.get())
            game_of_digits(color_point,level)
           
           
        l5play = tk.Button(win5_, text = "Ok",command = launch_game_of_digits)
        l5play.pack()
        l5play.place(x = 280, y = 138)
        
        root=tk.Frame (win5_, width=200, height=200)
        root.place(x=225,y=200)
        
        
        for i in range(len(l_scores)-1,-1,-1):
            b1=tk.Label(root, text="Level "+str(i+1)+":", font = ("Arial",13) ) 
            b1.grid(row=len(l_scores)-i-1, column=0)
            b1 = tk.Label(root, text=str(l_scores[i])+"%", font = ("Arial",13) )
            b1.grid(row=len(l_scores)-i-1, column=1)
    
       
    
    b5 = tk.Button(window, text = "Know more",command = gameOfDigits )
    b5.pack()
    b5.place(x=850,y=260)
    b5.config(width = 15,bg = 'light blue2')
    
    
    
    b5_ = tk.Button(window, text = "Play", command = gameOfDigitspl)
    b5_.pack()
    b5_.place(x=1000,y=260)
    b5_.config(width = 15,bg = 'lime green',fg = 'black')
    
    
    l6 = tk.Label(window, text = "4. Game Of Letter",font=("Arial ",20), bg= "pale turquoise")
    l6.place(x=300,y=300)
    
    def gameOfLetterspl():
        win6_ = tk.Tk()
        win6_.title("Begin Game Of Letters")
        win6_.geometry('600x300')  
        l6play = tk.Label(win6_ , text = "Select the color for pointer ", font = ("Arial",15),justify = CENTER)
        l6play.place(x=15,y=15)
        #print(l5play.winfo_width())
        l6playvar = tk.StringVar(win6_)
       
        choices = { 'Red','Blue','Green'}
        l6playvar.set('Blue') # set the default option
    
        popupMenu = tk.OptionMenu(win6_, l6playvar, *choices)
        popupMenu.place(x=475,y=15)
        
        l6play3 = tk.Label(win6_ , text = "Select the level you want to play", font = ("Arial",15),justify = CENTER)
        l6play3.place(x=15,y=40)
        l6playvar3 = tk.StringVar(win6_)
        
        inf_ob=Interface('gameofletters')
        l_scores=inf_ob.completed_levels()
        levels =[]
        for i in range(len(l_scores)+1,0,-1):
            levels.append(str(i))
        
        
        l6playvar3.set(levels[0])
        popupMenu3 = tk.OptionMenu(win6_, l6playvar3, *levels)
        popupMenu3.place(x=475,y=40)
       
        l6note = tk.Label(win6_, text = "Press 'q' to exit",font = ("Arial",15),justify = CENTER)
        l6note.place(x=220,y=100)
        
        name=inf_ob.getName()
        
        label = tk.Label(win6_ , text = "Levels completed by "+name+"-", font = ("Arial",15),justify = CENTER)
        label.place(x=120,y=170)     
        def launch_game_of_letters():
            global color_point
            color_point = str(l6playvar.get())
            level=str(l6playvar3.get())
            game_of_letters(color_point,level)
           
           
        l6play = tk.Button(win6_, text = "Ok",command = launch_game_of_letters)
        l6play.pack()
        l6play.place(x = 280, y = 138)
        
        root=tk.Frame (win6_, width=200, height=200)
        root.place(x=225,y=200)
        
        
        for i in range(len(l_scores)-1,-1,-1):
            b1=tk.Label(root, text="Level "+str(i+1)+":", font = ("Arial",13) ) 
            b1.grid(row=len(l_scores)-i-1, column=0)
            b1 = tk.Label(root, text=str(l_scores[i])+"%", font = ("Arial",13) )
            b1.grid(row=len(l_scores)-i-1, column=1)
       
    def gameOfLetter():
        win6 = tk.Tk()
        global video_name
        win6.title("Game Of Letter")
        win6.geometry('1400x750')
        l6_ = tk.Label(win6, text = l6text, font = ("Arial",15),justify = LEFT)
        l6_.place(x=15,y=50)
        video_name='gameofletter'
        b6_ = tk.Button(win6, text = "Watch Demonstration Video", command = play_demo )
        b6_.pack()
        b6_.place(x=550, y=650)
        b6_.config(width = 40,bg = 'green',fg = 'white')
    
    
    
    b6 = tk.Button(window, text = "Know more",command = gameOfLetter)
    b6.pack()
    b6.place(x=850,y=300)
    b6.config(width = 15,bg = 'light blue2')
    
    b6_ = tk.Button(window, text = "Play" , command = gameOfLetterspl)
    b6_.pack()
    b6_.place(x=1000,y=300)
    b6_.config(width = 15,bg = 'lime green',fg = 'black')
    
    
    
    l7 = tk.Label(window, text = "5. Game Of Color",font=("Arial ",20), bg= "pale turquoise")
    l7.place(x=300,y=340)
    
    
    def gameOfColorspl():
        win7_ = tk.Tk()
        win7_.title("Begin Game Of Colors")
        win7_.geometry('600x300')  
        l7play = tk.Label(win7_ , text = "Select the color for pointer ", font = ("Arial",15),justify = CENTER)
        l7play.place(x=15,y=15)
        #print(l5play.winfo_width())
        l7playvar = tk.StringVar(win7_)
       
        choices = { 'Red','Blue','Green'}
        l7playvar.set('Blue') # set the default option
    
        popupMenu = tk.OptionMenu(win7_, l7playvar, *choices)
        popupMenu.place(x=475,y=15)
        
        l7play3 = tk.Label(win7_ , text = "Select the level you want to play", font = ("Arial",15),justify = CENTER)
        l7play3.place(x=15,y=40)
        l7playvar3 = tk.StringVar(win7_)
        
        
        inf_ob=Interface('gameofcolors')
        l_scores=inf_ob.completed_levels()
        levels =[]
        for i in range(len(l_scores)+1,0,-1):
            levels.append(str(i))
        
        
        l7playvar3.set(levels[0])
        popupMenu3 = tk.OptionMenu(win7_, l7playvar3, *levels)
        popupMenu3.place(x=475,y=40)
       
        l7note = tk.Label(win7_, text = "Press 'q' to exit",font = ("Arial",15),justify = CENTER)
        l7note.place(x=220,y=100)
        
        name=inf_ob.getName()
        
        label = tk.Label(win7_ , text = "Levels completed by "+name+"-", font = ("Arial",15),justify = CENTER)
        label.place(x=120,y=170)     
       
        def launch_game_of_colors():
            global color_point
            color_point = str(l7playvar.get())
            level=str(l7playvar3.get())
            game_of_colors(color_point,level)
           
           
        l7play = tk.Button(win7_, text = "Ok",command = launch_game_of_colors)
        l7play.pack()
        l7play.place(x = 280, y = 138)
        
        root=tk.Frame (win7_, width=200, height=200)
        root.place(x=225,y=200)
        
        
        for i in range(len(l_scores)-1,-1,-1):
            b1=tk.Label(root, text="Level "+str(i+1)+":", font = ("Arial",13) ) 
            b1.grid(row=len(l_scores)-i-1, column=0)
            b1 = tk.Label(root, text=str(l_scores[i])+"%", font = ("Arial",13) )
            b1.grid(row=len(l_scores)-i-1, column=1)
       
       
       
    def gameOfColor():
        win7 = tk.Tk()
        global video_name
        win7.title("Game Of Color")
        win7.geometry('1400x750')
        l7_ = tk.Label(win7, text = l7text, font = ("Arial",15),justify = LEFT)
        l7_.place(x=15,y=50)
        video_name='gameofcolors'
        b7_ = tk.Button(win7, text = "Watch Demonstration Video", command = play_demo )
        b7_.pack()
        b7_.place(x=550, y=650)
        b7_.config(width = 40,bg = 'green',fg = 'white')
    
    b7 = tk.Button(window, text = "Know more",command = gameOfColor)
    b7.pack()
    b7.place(x=850,y=340)
    b7.config(width = 15,bg = 'light blue2')
    
    b7_ = tk.Button(window, text = "Play", command = gameOfColorspl)
    b7_.pack()
    b7_.place(x=1000,y=340)
    b7_.config(width = 15,bg = 'lime green',fg = 'black')
    
    
    l8 = tk.Label(window, text = "6. Game Of Shape",font=("Arial ",20), bg= "pale turquoise")
    l8.place(x=300,y=380)
    
    def gameOfShapespl():
        win8_ = tk.Tk()
        win8_.title("Begin Game Of Shapes")
        win8_.geometry('600x300')  
        l8play = tk.Label(win8_ , text = "Select the color for pointer ", font = ("Arial",15),justify = CENTER)
        l8play.place(x=15,y=15)
        #print(l5play.winfo_width())
        l8playvar = tk.StringVar(win8_)
       
        choices = { 'Red','Blue','Green'}
        l8playvar.set('Blue') # set the default option
    
        popupMenu = tk.OptionMenu(win8_, l8playvar, *choices)
        popupMenu.place(x=475,y=15)
        
        l8play3 = tk.Label(win8_ , text = "Select the level you want to play", font = ("Arial",15),justify = CENTER)
        l8play3.place(x=15,y=40)
        l8playvar3 = tk.StringVar(win8_)
        
        inf_ob=Interface('gameofshapes')
        l_scores=inf_ob.completed_levels()
        levels =[]
        for i in range(len(l_scores)+1,0,-1):
            levels.append(str(i))
        
        
        l8playvar3.set(levels[0])
        popupMenu3 = tk.OptionMenu(win8_, l8playvar3, *levels)
        popupMenu3.place(x=475,y=40)
       
        l8note = tk.Label(win8_, text = "Press 'q' to exit",font = ("Arial",15),justify = CENTER)
        l8note.place(x=220,y=100)
        
        name=inf_ob.getName()
        
        label = tk.Label(win8_ , text = "Levels completed by "+name+"-", font = ("Arial",15),justify = CENTER)
        label.place(x=120,y=170)     
        def launch_game_of_shapes():
            global color_point
            color_point = str(l8playvar.get())
            level=str(l8playvar3.get())
            game_of_shapes(color_point,level)
           
           
        l8play = tk.Button(win8_, text = "Ok",command = launch_game_of_shapes)
        l8play.pack()
        l8play.place(x = 280, y = 138)
       
        root=tk.Frame (win8_, width=200, height=200)
        root.place(x=225,y=200)
        
        
        for i in range(len(l_scores)-1,-1,-1):
            b1=tk.Label(root, text="Level "+str(i+1)+":", font = ("Arial",13) ) 
            b1.grid(row=len(l_scores)-i-1, column=0)
            b1 = tk.Label(root, text=str(l_scores[i])+"%", font = ("Arial",13) )
            b1.grid(row=len(l_scores)-i-1, column=1)
       
       
    
    def gameOfShape():
        win8 = tk.Tk()
        global video_name
        win8.title("Game Of Shape")
        win8.geometry('1400x750')
        l8_ = tk.Label(win8, text = l8text, font = ("Arial",15),justify = LEFT)
        l8_.place(x=15,y=50)
        video_name='gameofshapes'
        b8_ = tk.Button(win8, text = "Watch Demonstration Video", command = play_demo )
        b8_.pack()
        b8_.place(x=550, y=650)
        b8_.config(width = 40,bg = 'green',fg = 'white')
    
    
    b8 = tk.Button(window, text = "Know more", command = gameOfShape)
    b8.pack()
    b8.place(x=850,y=380)
    b8.config(width = 15,bg = 'light blue2')
    
    b8_ = tk.Button(window, text = "Play",command = gameOfShapespl)
    b8_.pack()
    b8_.place(x=1000,y=380)
    b8_.config(width = 15,bg = 'lime green',fg = 'black')
    
    
    l9 = tk.Label(window, text = "7. Game Of Number to Words",font=("Arial ",20), bg= "pale turquoise")
    l9.place(x=300,y=420)
    
    def gameOfNumber2Wordspl():
        win9_ = tk.Tk()
        win9_.title("Begin Game Of Number2Words")
        win9_.geometry('600x300')
        l9play1 = tk.Label(win9_ , text = "Select the color for pen", font = ("Arial",15),justify = CENTER)
        l9play1.place(x=15,y=15)
       
       
        l9play2 = tk.Label(win9_ , text = "Select the color for pointer", font = ("Arial",15),justify = CENTER)
        l9play2.place(x=15,y=40)
       
        l9playvar1 = tk.StringVar(win9_)
        l9playvar2 = tk.StringVar(win9_)
       
        choices = { 'Red','Blue','Green'}
        l9playvar1.set('Blue')
        l9playvar2.set('Red')
        # set the default option
    
        popupMenu1 = tk.OptionMenu(win9_, l9playvar1, *choices)
        popupMenu1.place(x=475,y=15)
       
        popupMenu2 = tk.OptionMenu(win9_, l9playvar2, *choices)
        popupMenu2.place(x=475,y=40)
        
        
       
        l9note = tk.Label(win9_, text = "Press 'q' to exit",font = ("Arial",15),justify = CENTER)
        l9note.place(x=220,y=100)
        
        
        def launch_game_of_num2words():
            global color_point,color_pen
            if str(l9playvar1.get())==str(l9playvar2.get()):
                showerror("Error", "color for pointer and color for pen can't be same.")
                return
            color_pen = str(l9playvar1.get())
            color_point = str(l9playvar2.get())
            game_of_num2words(color_point,color_pen)
           
           
        l9play = tk.Button(win9_, text = "Ok",command = launch_game_of_num2words)
        l9play.pack()
        l9play.place(x = 280, y = 138)
        
        root=tk.Frame (win9_, width=200, height=200)
        root.place(x=225,y=200)
        
    def GameOfNumber2Words():
        win9 = tk.Tk()
        global video_name
        win9.title("Game of Number2Words")
        win9.geometry('1400x750')
        l9_ = tk.Label(win9, text = l9text, font = ("Arial",15),justify = LEFT)
        l9_.place(x=15,y=50)
        video_name='gameofnum2words'
        b9_ = tk.Button(win9, text = "Watch Demonstration Video", command = play_demo )
        b9_.pack()
        b9_.place(x=550, y=650)
        b9_.config(width = 40,bg = 'green',fg = 'white')
       
    b9 = tk.Button(window, text = "Know more", command = GameOfNumber2Words)
    b9.pack()
    b9.place(x=850,y=420)
    b9.config(width = 15,bg = 'light blue2')
    
    b9_ = tk.Button(window, text = "Play", command = gameOfNumber2Wordspl )
    b9_.pack()
    b9_.place(x=1000,y=420)
    b9_.config(width = 15,bg = 'lime green',fg = 'black')
    
    
    
    
    l10 = tk.Label(window, text = "8. E-sign",font=("Arial ",20), bg= "pale turquoise")
    l10.place(x=300,y=460)
    
    def gameOfesignpl():
        win10_ = tk.Tk()
        win10_.title("Begin")
        win10_.geometry('600x300')
        l10play1 = tk.Label(win10_ , text = "Select the color for pen", font = ("Arial",15),justify = CENTER)
        l10play1.place(x=15,y=15)
       
       
        l10play2 = tk.Label(win10_ , text = "Select the color for pointer", font = ("Arial",15),justify = CENTER)
        l10play2.place(x=15,y=40)
       
        l10playvar1 = tk.StringVar(win10_)
        l10playvar2 = tk.StringVar(win10_)
       
        choices = { 'Red','Blue','Green'}
        l10playvar1.set('Blue')
        l10playvar2.set('Red')
        # set the default option
    
        popupMenu1 = tk.OptionMenu(win10_, l10playvar1, *choices)
        popupMenu1.place(x=475,y=15)
       
        popupMenu2 = tk.OptionMenu(win10_, l10playvar2, *choices)
        popupMenu2.place(x=475,y=40)
        
        
        l10note = tk.Label(win10_, text = "Press 'q' to exit",font = ("Arial",15),justify = CENTER)
        l10note.place(x=220,y=100)
        
        
        def launch_game_of_esigns():
            global color_point,color_pen
            if str(l10playvar1.get())==str(l10playvar2.get()):
                showerror("Error", "color for pointer and color for pen can't be same.")
                return
            color_pen = str(l10playvar1.get())
            color_point = str(l10playvar2.get())
            eSign(color_point,color_pen)
           
           
        l10play = tk.Button(win10_, text = "Ok",command = launch_game_of_esigns)
        l10play.pack()
        l10play.place(x = 280, y = 138)
        
        root=tk.Frame (win10_, width=200, height=200)
        root.place(x=225,y=200)
        
    def Esign():
        win10 = tk.Tk()
        win10.title("eSign")
        global video_name
        win10.geometry('1400x750')
        l10_ = tk.Label(win10, text = l10text, font = ("Arial",15),justify = LEFT)
        l10_.place(x=15,y=50)
        video_name='esign'
        b10_ = tk.Button(win10, text = "Watch Demonstration Video", command = play_demo )
        b10_.pack()
        b10_.place(x=550, y=650)
        b10_.config(width = 40,bg = 'green',fg = 'white')
       
    
    
    b10 = tk.Button(window, text = "Know more", command = Esign)
    b10.pack()
    b10.place(x=850,y=460)
    b10.config(width = 15,bg = 'light blue2')
    
    b10_ = tk.Button(window, text = "Play",command = gameOfesignpl)
    b10_.pack()
    b10_.place(x=1000,y=460)
    b10_.config(width = 15,bg = 'lime green',fg = 'black')
    
    def howtoplay():
        win11 = tk.Tk()
        win11.title("Instructions")
        win11.geometry('1400x750')
        l11_ = tk.Label(win11, text = l11text, font = ("Arial",15),justify = LEFT)
        l11_.place(x=15,y=50)
        
        
    l10_ = tk.Label(window, text = "For more instructions on how to play, click here", font = ("Arial",15), bg= "pale turquoise")
    l10_.place(x=420,y=600)
       
    b11 = tk.Button(window, text = "Instructions", font = ("Arial",10), command = howtoplay)
    b11.pack()
    b11.place(x=845,y=600)
    b11.config(width = 15,bg = 'light blue2')
    
    
    
    window.mainloop()
    
