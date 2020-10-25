# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:24:00 2020

@author: kmiao
"""

#--------------------------Kay Part----------------------------
y_can.shape
new_max=np.zeros((275,5))
#new_df=pd.DataFrame(columns=("1", "2", "3", "4", "5"))
for i in range(0, 275):
    new_max[i]=y_can[i:(i+5), 0]

new_max_one = np.hstack((np.ones((275, 1)), new_max))