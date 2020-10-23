# standard Python imports
import os
import argparse
import time
import pickle


# 3rd party libraries
import numpy as np                              # this comes with Anaconda
import pandas as pd                             # this comes with Anaconda
import matplotlib.pyplot as plt                 # this comes with Anaconda
from sklearn.tree import DecisionTreeClassifier # if using Anaconda, install with `conda install scikit-learn`
from sklearn.naive_bayes import GaussianNB
from numpy.linalg import norm
from numpy import dot


# CPSC 340 code
import utils
# from random_forest import DecisionStumpErrorRate, DecisionStumpGiniIndex, RandomForest
# from naive_bayes import  NaiveBayes
# from knn import KNN    
# from stacking import Stacking


if __name__ == "__main__":
    with open(os.path.join('..','data','wordvec_test.csv'), 'rb') as f:
        data = pd.read_csv(f, header = 0).to_numpy()
    
    # X = 
    # y = 









