import numpy as np


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



class alpine2:
    '''
    Alpine2 function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, input_dim, bounds=None):
        if bounds is None:
            self.bounds = [(1, 10)] * input_dim
        else:
            self.bounds = bounds
        self.min = [(7.917)] * input_dim
        self.fmin = -2.808 ** input_dim
        self.input_dim = input_dim

    def f(self, X):
        X = np.array(X).reshape(1, -1).T
        print(X)
        n = X.shape[0]
        fval = np.cumprod(np.sqrt(X), axis=1)[:, self.input_dim - 1] * np.cumprod(np.sin(X), axis=1)[:,
                                                                       self.input_dim - 1]
        if self.sd == 0:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, self.sd, n).reshape(n, 1)
        return -fval.reshape(n, 1) + noise

class ackley:
    '''
    Ackley function
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, input_dim, bounds=None):
        self.input_dim = input_dim

        if bounds == None:
            self.bounds = [(-32.768, 32.768)] * self.input_dim

        else:
            self.bounds = bounds

        self.min = [(0) * self.input_dim]

        self.fmin = 0

    def f(self, X):

        X = np.array(X).reshape(1, -1).T
        print('X:', X)
        n = X.shape[0]
        fval = (X * np.sin(X) + 0.1 * X).sum(axis=1)
        return fval.reshape(n, 1)


class Test_func01:
    '''
    Ackley function
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, input_dim, bounds=None):
        self.input_dim = input_dim

        if bounds == None:
            self.bounds = [[2.1, 9.9]] * self.input_dim

        else:
            self.bounds = bounds

        # self.min = [(0) * self.input_dim]
#
        # self.fmin = 0

    def f(self, X):

        X = np.array(X).reshape(1, self.input_dim).T


        One = (np.log10(X - 2) ** 2).sum()

        Two = (np.log10(10 - X) ** 2).sum()

        Three = 1.0
        for i in range(self.input_dim):
            Three *= X[i]
        fval = One + Two + Three[0]
        return fval

class xinba:

    def __init__(self, input_dim=10, bounds=None):
        self.input_dim = input_dim

        if bounds == None:
            self.bounds = [[-1, 1]] * self.input_dim
        else:
            self.bounds = bounds

        self.min = [0] * self.input_dim
        self.fmin = 0
        self.name = 'x2 ** 2 + x1 * x3 + x1 - 4'


    def f(self, X):

        # X = np.array(X).reshape(1, self.input_dim)
        f_value = 0
        for i in range(self.input_dim):
            f_value += abs(X[i] * np.sin(X[i]) + 0.1 * X[i])
        return f_value