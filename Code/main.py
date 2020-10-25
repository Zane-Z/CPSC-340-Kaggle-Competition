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
import knn
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
        with open(os.path.join('..','data','phase1_training_data.csv'), 'rb') as f:
            data1 = pd.read_csv(f, header = 0)


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
        model.fit(X_world_cases, y_world)
        y_pred = model.predict(X_world_cases)
        utils.test_and_plot(model,X_world_cases,y_world,Xtest=None,ytest=None,title="World",filename="World_cases_feature.pdf")

        
        def calculate_poly_w (X, y):
            model = linear_model.LeastSquaresPoly(p = 5)
            model.fit(X, y)
            w = model.print_w()
            return w


        print("Poly_Canadian: ")
        model = linear_model.LeastSquaresPoly(p = 5)
        model.fit(X_can_cases, y_can)
        y_pred = model.predict(X_can_cases)
        w_can_case = calculate_poly_w(X_can_cases, y_can)
        w_can_case_100k = calculate_poly_w(X_can_cases_100k, y_can)
        utils.test_and_plot(model,X_can_cases,y_can,Xtest=None,ytest=None,title="Canadian Poly",filename="Canadian_cases_feature_poly.pdf")
        

        # Finding distance and use that to find the cloest country to canada based on the euclidean distance of their poly model w. 
        X_name_unique = data1.country_id.unique()
        country_w_case = np.zeros((len(X_name_unique), 6))
        country_w_100k = np.zeros((len(X_name_unique), 6))
        count = 0
        # print(X_name_unique[137])
        for i in range(len(X_name_unique)):
            X_country = data[data[:, 0] == X_name_unique[i]]
            X_country_case = X_country[:, 2]
            X_country_case = np.reshape(X_country_case, (X_country_case.shape[0], 1))
            X_country_case = X_country_case.astype(float)
            # print(np.shape(X_country_case))
            X_country_100k = X_country[:, 5]
            X_country_100k = np.reshape(X_country_100k, (X_country_100k.shape[0], 1))
            X_country_100k = X_country_100k.astype(float)
            y_country = X_country[:, 3]
            y_country = np.reshape(y_country, (y_country.shape[0], 1))
            y_country = y_country.astype(float)
            # print(np.shape(y_country))

            # print(count)
            w_country_case = calculate_poly_w(X_country_case, y_country).T
            # print(np.shape(w_country_case))
            w_country_100k = calculate_poly_w(X_country_100k, y_country).T
            # print(np.shape(w_country_100k))
            country_w_case[i, :] = w_country_case
            country_w_100k[i, :] = w_country_100k
            count += 1

        dist_case = utils.euclidean_dist_squared(w_can_case.T, country_w_case)
        dist_100k = utils.euclidean_dist_squared(w_can_case_100k.T, country_w_100k)

        dist = dist_case + dist_100k
        print(dist)


        # TODO: 1. Find the most similar countries to Can by sorting dist
        # 2. combine their data and fit a poly model 
        # 3. compare with the can only poly model's error to decide which works better
        # 4. try using Death counts to find the most similar country





        # TODO:
        # model1 = linear_model.LeastSquaresPoly(p = 5)

        # model1.fit(X_can_cases_14_100k, y_can)

        # utils.test_and_plot(model1,X_can_cases_14_100k,y_can,Xtest=None,ytest=None,title="Canadian Poly",filename="Canadian_14_100k_feature_poly.pdf")


















