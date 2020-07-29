# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:10:19 2019

@author: hp
"""

from tkinter import *
from tkinter.messagebox import *
import tkinter as tk
from signup import runSignUp
from home import start
import pandas as pd
import os
#start()
# create a GUI window 
root = tk.Tk() 
  
    # set the background colour of GUI window 
root.configure(background='pale turquoise') 
  
    # set the title of GUI window 
root.title("Login Page") 
  
    # set the configuration of GUI window 
root.geometry("500x250") 
 
    # create a Form label 
heading = tk.Label(root, text="Fill the credentials", bg="pale turquoise", font = ("Arial",15)) 
  
    # create a Name label 
Username = tk.Label(root, text="User Id", bg="pale turquoise", font = ("Arial",10)) 
  
    # create a Course label 
Password = tk.Label(root, text="Password", bg="pale turquoise", font = ("Arial",10)) 


heading.place(x=160, y=40)
Username.place(x= 140, y= 80)
Password.place(x= 140, y= 100)
  
 
  
    # create a text entry box 
    # for typing the information 
Username_field = Entry(root) 
Password_field = Entry(root, show= "•") 

Username_field.place(x= 220, y= 80)
Password_field.place(x= 220, y= 100)
 
   


def checklogin():
    dfl=pd.read_csv('login.csv')
    dfm=dfl.iloc[:,:].values
    user_id=str(Username_field.get())
    pwd=str(Password_field.get())
    if user_id.isnumeric() and user_id[0]!='-' and int(user_id)<dfm.shape[0] and str(dfm[int(user_id)][0])==pwd: 
        cnfg=pd.read_csv('configure.csv')
        os.remove('configure.csv')
        cnfg.loc[0,'uid']=user_id
        cnfg.to_csv('configure.csv',index=False)
        root.destroy()
        start()
    else:
        showerror("Error", "Invalid user_id/password")
    # create a Submit Button and place into the root window 
login = tk.Button(root, text="Login", fg="Black", bg="light blue", command= checklogin) 
login.place(x=235, y=125)

signUp = tk.Label(root, text="Don't have an account? Sign up here:", bg="pale turquoise", font = ("Arial",10))
signUpButton = tk.Button(root, text="Sign up", fg="Black", bg="light blue", command= runSignUp )   

signUp.place(x=110, y=210)
signUpButton.place(x=335, y=208)

    # start the GUI 
root.mainloop() 