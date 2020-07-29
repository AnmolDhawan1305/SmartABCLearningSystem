# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:52:11 2019

@author: hp
"""

from tkinter import *
from tkinter.messagebox import *
import tkinter as tk
import pandas as pd


def runSignUp():
    # create a GUI window 
    root = tk.Tk() 
  
    # set the background colour of GUI window 
    root.configure(background='pale turquoise') 
  
    # set the title of GUI window 
    root.title("Sign Up Page") 
  
    # set the configuration of GUI window 
    root.geometry("500x250") 
 
    # create a Form label 
    heading = tk.Label(root, text="Fill the credentials", bg="pale turquoise", font = ("Arial",15)) 
  
    # create a Name label 
    Fname = tk.Label(root, text="First Name", bg="pale turquoise", font = ("Arial",10)) 
  
    # create a Course label 
    Lname = tk.Label(root, text="Last Name", bg="pale turquoise", font = ("Arial",10)) 
  
    # create a Semester label 
    Class = tk.Label(root, text="Class", bg="pale turquoise", font = ("Arial",10)) 
  
    # create a Form No. lable 
    password = tk.Label(root, text="Password", bg="pale turquoise", font = ("Arial",10)) 
  
    # create a Contact No. label 
    rpassword = tk.Label(root, text="Retype Password", bg="pale turquoise", font = ("Arial",10)) 
  
    # grid method is used for placing 
   # the widgets at respective positions 
    # in table like structure . 
    heading.grid(row=0, column=1) 
    Fname.grid(row=1, column=0) 
    Lname.grid(row=2, column=0) 
    Class.grid(row=3, column=0) 
    password.grid(row=4, column=0) 
    rpassword.grid(row=5, column=0) 
  
    # create a text entry box 
    # for typing the information 
    Fname_field = Entry(root) 
    Lname_field = Entry(root) 
    Class_field = Entry(root) 
    password_field = Entry(root, show="•") 
    rpassword_field = Entry(root, show="•") 
 
  
    # grid method is used for placing 
    # the widgets at respective positions 
    # in table like structure . 
    Fname_field.grid(row=1, column=1, ipadx="100") 
    Lname_field.grid(row=2, column=1, ipadx="100") 
    Class_field.grid(row=3, column=1, ipadx="100") 
    password_field.grid(row=4, column=1, ipadx="100") 
    rpassword_field.grid(row=5, column=1, ipadx="100")  
    
    def do_signup():
        fname=str(Fname_field.get())
        lname=str(Lname_field.get())
        Class=str(Class_field.get())
        pwd=str(password_field.get())
        pwd_r=str(rpassword_field.get())
        if not Class.isnumeric():
            showerror("Error", "Class has to be a number.")
        elif pwd!=pwd_r:
            showerror("Error", "Passwords do not match")
        else:
            data = [[pwd, fname, lname, Class]] 
            df = pd.DataFrame(data)
            df.to_csv('login.csv', mode='a', header=False, index=False)
            cnf_df=pd.read_csv('configure.csv')
            r=cnf_df.iloc[-1][2]
            scores_df=pd.read_csv('scores.csv')
            uid=scores_df.shape[0]/r
            data=[]
            for i in range(r):
                temp=[0]
                data.append(temp)
            df = pd.DataFrame(data)
            df.to_csv('scores.csv', mode='a', header=False, index=False)
            showinfo("Remember", "Your User Id is: "+str(int(uid))+"\nNote: Remember the above User Id for login.")
            root.destroy()
    # create a Submit Button and place into the root window 
    submit = tk.Button(root, text="Sign Up", fg="Black", bg="light blue", command= do_signup) 
    submit.place(x=240, y=140)
  
    # start the GUI 
    root.mainloop()
    