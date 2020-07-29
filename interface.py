# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:10:55 2019

@author: HP 250 G5
"""

import pandas as pd
import os

class Interface:
    def __init__(self,game_name):
        self.game_name=game_name
        cnfg=pd.read_csv('configure.csv')
        cnfg=cnfg.iloc[:,:].values
        self.cnfg=cnfg
        game_no=0
        while game_no<int(cnfg[-1][1]):
            if str(cnfg[game_no][1]).lower()==game_name.lower(): break
            game_no=game_no+1
        self.game_number=game_no
        
        
    def getMaxLevel(self):
        return int(self.cnfg[self.game_number][2])
    
    def update_scores(self,level,score):
        uid=int(self.cnfg[0][0])
        sum_l=0
        for i in range(self.game_number):
            sum_l=sum_l+self.cnfg[i][2]
            
        idx=uid*int(self.cnfg[-1][2])+sum_l+(level-1)
        d_scores=pd.read_csv('scores.csv')
        os.remove('scores.csv')
        d_scores.iloc[idx][0]=score*100
        d_scores.to_csv('scores.csv',index=False)
        
    def completed_levels(self):
        l=[]
        uid=int(self.cnfg[0][0])
        sum_l=0
        for i in range(self.game_number):
            sum_l=sum_l+self.cnfg[i][2]
            
        idx=uid*int(self.cnfg[-1][2])+sum_l
        n=int(self.cnfg[self.game_number][2])
        d_scores=pd.read_csv('scores.csv')
        for i in range(n):
            temp=d_scores.iloc[idx+i][0]
            if temp==0: break
            l.append(temp)
        return l
        
    def getName(self):
        df=pd.read_csv('login.csv')
        return str(df.iloc[int(self.cnfg[0][0])][1]).capitalize()+' '+str(df.iloc[int(self.cnfg[0][0])][2]).capitalize()