import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        self.w = solve(X.T@(z*X), X.T@(z*y))

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w.flatten(), lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        ''' MODIFY THIS CODE '''

        if w.ndim is 1:
   	        w = w[:, np.newaxis]

        # Calculate the function value
        f = np.sum(np.log(np.exp(X@w - y) + np.exp(y - X@w)))

        # Calculate the gradient value
        g = np.zeros((1, X.shape[1]))
        
        for i in range(X.shape[1]):
            summ = 0
            for n in range(X.shape[0]):
                summ = np.sum(X[n, 0]*(np.exp(w.T*X[n, 0] - y[n, 0]) - np.exp(y[n, 0] - w.T*X[n,0]))/(np.exp(w.T*X[n,0] - y[n,0]) + np.exp(y[n,0] - w.T*X[n,0])))
            g[i] = summ
        return (f,g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        w_0 = np.append(np.ones((X.shape[0], 1)), X, axis = 1)
        self.w = solve(w_0.T@w_0, w_0.T@y)

    def predict(self, X):
        w_0 = np.append(np.ones((X.shape[0],1)), X, axis=1)
        return w_0@self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        Z = self.__polyBasis(X)
        self.w = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        Z = self.__polyBasis(X)
        return Z@self.w

    def print_w(self):
        return self.w

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        Z = np.ones((X.shape[0], self.p + 1))
        for i in range(1, self.p + 1):
            Z[:, i] = X[:, 0] ** i
        return Z