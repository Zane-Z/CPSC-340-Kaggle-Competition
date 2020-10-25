# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:24:00 2020

@author: kmiao
"""
import numpy as np                              # this comes with Anaconda
import pandas as pd                             # this comes with Anaconda

#--------------------------Kay Part----------------------------
class TimeSeries:
    def __init__(self):
        pass
    
    def get_tseries_X(self, X, window_length=5, preapp_one=True): 
        x_shape = X.shape
        num_row = x_shape[0]
        num_col = x_shape[1]
        new_dim = num_row-window_length
    
        if (num_col==1):
            new_max=np.zeros((new_dim, window_length))
    
            for i in range(0, new_dim):
                new_max[i]=X[i:(i+window_length), 0]
            if (preapp_one == True):
                new_max = np.hstack((np.ones((new_dim, 1)), new_max))
              
            return new_max
            
        elif (num_col>1):
            return "invalid input"
        else:
            return "invalid input"
        
        
    def get_tseries_Y(self, Y, window_length=5):    
        new_y = Y[window_length:,]
        return new_y    