"""
Implementation of k-nearest neighbours classifier with euclidean_dist_square
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        squared_dist = utils.euclidean_dist_squared(Xtest, self.X)
        # sorted_index = np.argsort(squared_dist)
        output = np.zeros(Xtest.shape[0])

        for i in range(squared_dist.shape[0]):
            sorted_index = np.argsort(squared_dist[i, :])
            arr = np.zeros(self.k)
            for j in range(self.k):
                n = sorted_index[j]
                arr[j] = self.y[n]
            output[i] = stats.mode(arr)[0]

        return output