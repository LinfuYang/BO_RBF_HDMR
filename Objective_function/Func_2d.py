import numpy as np

class Bohachevsky:

    def __init__(self, bounds=None):

        if bounds == None:
            self.bounds = [[-1, 1]] * 2
        self.min = [0, -0.24]
        self.fmin = -0.24

    def f(self, x):
        '''
        :param x: 一维数组
        :return:  函数值
        '''

        x1 = x[0]
        x2 = x[1]

        f_value = x1 ** 2 + x2 ** 2 - 0.3 * np.cos(3 * np.pi * x1) + 0.3 * np. cos(4 * np.pi * x2) + 0.3
        return f_value


class branin:
    '''
    Branin function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, a=None, b=None, c=None, d=None, e=None, r=None):
        self.input_dim = 2
        if bounds == None:
            self.bounds = [[-5, 10], [0, 15]]

        if a == None: self.a = 1
        else: self.a = a
        if b == None: self.b = 5.1/(4*np.pi**2)
        else: self.b = b
        if c == None: self.c = 5/np.pi
        else: self.c = c
        if d == None: self.d = 6
        else: self.d = d
        if e == None: self.e = 10
        else: self.e = e
        if r == None: self.r = 1 / (8 * np.pi)
        else: self.r = r

        self.min = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]
        self.fmin = 0.397887
        self.name = 'Branin'

    def f(self, Y):

        X = np.array(Y).reshape(1, self.input_dim)

        x1 = X[:, 0]
        x2 = X[:, 1]

        fval = self.a * (x2 - self.b * x1 ** 2 + self.c * x1 - self.d) ** 2 + self.e * (1 - self.r) * np.cos(
            x1) + self.e
        return fval[0]
