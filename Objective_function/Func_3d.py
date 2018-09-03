import numpy as np


class lizi_3d:
    '''

    书中例子，
    '''
    def __init__(self, bounds=None):
        '''
        :param bounds: 函数各维度的取值区间，用二维数组表示
        '''

        self.input_dim = 3
        if bounds is None:
            self.bounds = [[0, 1], [0, 1], [0, 1]]
        else:
            self.bounds = bounds
        self.min = [0, 0, 0]
        self.fmin = -4
        self.name = 'x2 ** 2 + x1 * x3 + x1 - 4'

    def f(self, x):
        '''
        :param X: 一维数组，具体的函数采样点
        :return: 具体的函数值
        '''
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return x2 ** 2 + x1 * x3 + x1 - 4

class sin_3d:

    def __init__(self, bounds=None):
        ''''
        :param
        bounds: 函数各维度的取值区间，用二维数组表示
        '''

        self.input_dim = 3
        if bounds is None:
            self.bounds = [[-1, 1]] * 3
        else:
            self.bounds = bounds
        self.min = [0, 0, 0]
        self.fmin = 0
        self.name = 'x2 ** 2 + x1 * x3 + x1 - 4'


    def f(self, x):
        '''
        :param X: 一维数组，具体的函数采样点
        :return: 具体的函数值
        '''
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        # x4 = x[3]
        # x5 = x[4]
        # x6 = x[5]
        # x7 = x[6]
        # x8 = x[7]
        # x9 = x[8]
        # x10 = x[9]

        return abs(x1 * np.sin(x1) + 0.1 * x1) + \
               abs(x2 * np.sin(x2) + 0.1 * x2) + \
               abs(x3 * np.sin(x3) + 0.1 * x3)
               # abs(x4 * np.sin(x4) + 0.1 * x4) + \
               # abs(x5 * np.sin(x5) + 0.1 * x5) + \
               # abs(x6 * np.sin(x6) + 0.1 * x6) + \
               # abs(x7 * np.sin(x7) + 0.1 * x7) + \
               # abs(x8 * np.sin(x8) + 0.1 * x8) + \
               # abs(x9 * np.sin(x9) + 0.1 * x9) + \
               # abs(x10 * np.sin(x10) + 0.1 * x10)


class x1x2x3:
    '''

    书中例子，
    '''
    def __init__(self, bounds=None):
        '''
        :param bounds: 函数各维度的取值区间，用二维数组表示
        '''

        self.input_dim = 3
        if bounds is None:
            self.bounds = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
        else:
            self.bounds = bounds
        self.min = [0, 0, 0]
        self.fmin = -1
        self.name = 'x2 ** 2 + x1 * x3 + x1 - 4'

    def f(self, x):
        '''
        :param X: 一维数组，具体的函数采样点
        :return: 具体的函数值
        '''
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        return x1 * x2 * x3 * x4 * x5 * x6