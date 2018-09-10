import numpy as np
import math

class alpine1:
    '''
    Alpine1 function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, input_dim, bounds=None):
        if bounds is None:
            self.bounds = [[-10, 10]] * input_dim
        else:
            self.bounds = bounds
        self.min = [(0)] * input_dim
        self.fmin = 0
        self.input_dim = input_dim

    def f(self, X):
        '''
        :param X: 一位数组
        :return:
        '''
        X = np.array(X).reshape(1, -1)

        fval = (X * np.sin(X) + 0.1 * X).sum(axis=1)


        return fval[0]


class xinba:

    def __init__(self, input_dim=10, bounds=None):
        self.input_dim = input_dim

        if bounds == None:
            self.bounds = [[-1, 1]] * self.input_dim
        else:
            self.bounds = bounds

        self.min = [-0.05] * self.input_dim
        self.fmin = -0.0025 * self.input_dim
        self.name = 'x2 ** 2 + x1 * x3 + x1 - 4'

    def f(self, X):

        # X = np.array(X).reshape(1, self.input_dim)
        f_value = 0
        for i in range(self.input_dim):
            f_value += (X[i] * np.sin(X[i]) + 0.1 * X[i])
        return f_value


class xxxx:
    def __init__(self, input_dim=10, bounds=None):
        self.input_dim = input_dim

        if bounds == None:
            self.bounds = [[-1, 1]] * self.input_dim
        else:
            self.bounds = bounds

        self.min = [0] * self.input_dim
        self.fmin = 0
        self.name = 'x1 *... xn'

    def f(self, X):

        # X = np.array(X).reshape(1, self.input_dim)
        f_value = X[0]
        for i in range(1, self.input_dim):
            f_value = f_value * X[i]
        return f_value
from numpy.linalg import *
class  Gaussian_mixture_function:


    def __init__(self, input_dim=None, bounds=None, mu1=None,mu2=None,sigma1=None,sigma2=None):
        # 维度
        if input_dim != None:
            self.input_dim = input_dim
        else:
            self.input_dim = 10
        # 取值区间
        if bounds != None:
            self.bounds = bounds
        else:
            self.bounds = [[1, 4]] * self.input_dim
        # mu1
        if mu1 != None:
            self.mu1 = mu1
        else:
            self.mu1 = np.mat([2] * self.input_dim)
        # mu2
        if mu2 != None:
            self.mu2 = mu2
        else:
            self.mu2 = np.mat([3] * self.input_dim)
        # sigma1
        if sigma1 != None:
            self.sigma1 = sigma1
        else:
            self.sigma1 = np.mat(np.eye(self.input_dim))
        # sigma2
        if sigma2 != None:
            self.sigma2 = sigma2

        else:
            self.sigma2 = np.mat(np.eye(self.input_dim))
            # print('self.sigma2.I:', self.sigma2.I)
        self.func_name = 'Gaussian_mixture_function'
        self.x_min = [2] * self.input_dim

    def f(self, X):
        '''
        :param X:
        :return:
        '''

        X = np.mat(X)

        np_1_one = ((2 * np.pi) ** (self.input_dim / 2)) * (det(self.sigma1) ** (1 / 2))
        np_2_one = ((2 * np.pi) ** (self.input_dim / 2)) * (det(self.sigma2) ** (1 / 2))

        np_1_two = np.exp((-1 / 2) * ((X - self.mu1) * (self.sigma1.I) * (X - self.mu1).T))
        np_2_two = np.exp((-1 / 2) * ((X - self.mu2) * (self.sigma2.I) * (X - self.mu2).T))
        # print('self.sigma2.I:', self.sigma2.I)


        f_value = (1 / np_1_one) * np_1_two + 0.5 * (1 / np_2_one) * np_2_two

        return -f_value[0, 0] * 100



class Schwefel_Func:

    def __init__(self, input_dim=None, bounds=None):

        if input_dim == None:
            self.input_dim = 10
        else:
            self.input_dim = input_dim
        if bounds == None:
            self.bounds = [[-1, 1]] * self.input_dim
        else:
            self.bounds = bounds

    def f(self, X):
        f_1 = 0
        for j in range(self.input_dim):
            f_2 = 0
            for i in range(0, j + 1):
                    f_2 += X[i]
            f_1 += f_2 ** 2

        return f_1





