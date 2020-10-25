# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:24:00 2020

@author: kmiao
"""

import numpy as np
from decision_tree import DecisionTree
from decision_stump import DecisionStumpErrorRate

class Emsemble:

    def __init__(self, max_depth, models, stump_class=DecisionStumpErrorRate):
        self.max_depth = max_depth
        self.stump_class = stump_class
        self.models = models
        self.tree_mod = DecisionTree(max_depth, stump_class)

    def fit(self, X, y):
        # Fits a decision tree using greedy recursive splitting
        N, D = X.shape
        L=len(self.models)
        
        Xnew = np.zeros((N, L))
        cur_col_ind = 0
        for model in self.models:
            y = model.predict(X)
            Xnew[:, cur_col_ind] = y
            cur_col_ind = cur_col_ind
        
        self.tree_mod.fit(Xnew, y, self.max_depth, self.stump_class)
        
        
    def predict(self, X):
        
        y= self.tree_mod.pred(X)

        return y
            
            
    
        
