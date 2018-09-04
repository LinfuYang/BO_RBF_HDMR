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
        f_value = 1
        for i in range(self.input_dim):
            f_value += f_value * X[i]
        return f_value