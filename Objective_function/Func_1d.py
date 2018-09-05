import numpy as np

class test_1d_func:
    def __init__(self, bounds=None):

        if bounds == None:
            self.bounds = [[-1, 1]]
        self.min = [-0.05]
        self.fmin = -0.0025
        self.input_dim = 1
    def f(self, XY):
        '''
        :param XY: 一维数组
        :return:  函数值
        '''

        x = XY[0]
        return x * np.sin(x) + 0.1 * x