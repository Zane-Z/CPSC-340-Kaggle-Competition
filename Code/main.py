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
import linear_model
from utils2 import TimeSeries
# from random_forest import DecisionStumpErrorRate, DecisionStumpGiniIndex, RandomForest
# from naive_bayes import  NaiveBayes
# from knn import KNN    
# from stacking import Stacking


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c','--case', required=True)
    # io_args = parser.parse_args()
    # case = io_args.case

    if 1:
        
        print("Canadian: ")
        with open(os.path.join('..','data','phase1_training_data.csv'), 'rb') as f:
            data = pd.read_csv(f, header = 0).to_numpy()
    
        X_all = data[data[:, 0] == "CA"]
        y_can = X_all[:, 3]
        y_can = np.reshape(y_can, (y_can.shape[0], 1))
        y_can = y_can.astype(float)
        #retrieve data thats canadian alone
        def get_feature(X, col_name):
            X = X_all[:, col_name]
            X = np.reshape(X, (X.shape[0], 1))
            X = X.astype(float)

            return X

        X_can_all = np.delete(X_all, [0, 1, 3], axis = 1)
        X_can_all = np.reshape(X_can_all, (X_can_all.shape[0], 3))
        X_can_all = X_can_all.astype(float)

        X_can_cases = get_feature(X_all, 2)
        X_can_cases_14_100k = get_feature(X_all, 4)
        X_can_cases_100k = get_feature(X_all, 5)

        model = linear_model.LeastSquares()
        w = model.fit(X_can_cases, y_can)
        y_pred = model.predict(X_can_cases)
        utils.test_and_plot(model,X_can_cases,y_can,Xtest=None,ytest=None,title="Canadian",filename="Canadian_cases_feature.pdf")


        # model.fit(X_cases, y)
        # y_pred = model.predict(X_cases)


        # model.fit(X_cases, y)
        # y_pred = model.predict(X_cases)



        # y_pred = model.predict(X_test)
        # # y_pred = model.predict(X_cases)



        print("World: ")

        X_world = data

        y_world = X_world[:, 3]
        y_world = np.reshape(y_world, (y_world.shape[0], 1))
        y_world = y_world.astype(float)

        X_world_cases = X_world[:, 2]
        X_world_cases = np.reshape(X_world_cases, (X_world_cases.shape[0], 1))
        X_world_cases = X_world_cases.astype(float)

        # print(np.shape(X_world_cases))
        # print(np.shape(y_world))


        model = linear_model.LeastSquares()
        w = model.fit(X_world_cases, y_world)
        y_pred = model.predict(X_world_cases)
        utils.test_and_plot(model,X_world_cases,y_world,Xtest=None,ytest=None,title="World",filename="World_cases_feature.pdf")


        print("Poly_Canadian: ")
        model = linear_model.LeastSquaresPoly(p = 5)
        w = model.fit(X_can_cases, y_can)
        y_pred = model.predict(X_can_cases)

        utils.test_and_plot(model,X_can_cases,y_can,Xtest=None,ytest=None,title="Canadian Poly",filename="Canadian_cases_feature_poly.pdf")
        

        print(f["country_id"].unique())


        model1 = linear_model.LeastSquaresPoly(p = 5)

        model1.fit(X_can_cases_14_100k, y_can)

        utils.test_and_plot(model1,X_can_cases_14_100k,y_can,Xtest=None,ytest=None,title="Canadian Poly",filename="Canadian_14_100k_feature_poly.pdf")



#--------------------------Kay Part----------------------------
ts_model=TimeSeries()
#ts_model.matrix_to_tseries(y_can, y_can)
new_X_from_function_3 = ts_model.get_tseries_X(y_can)
new_Y_from_function_3 = ts_model.get_tseries_Y(y_can)












