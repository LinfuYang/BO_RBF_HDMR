


class lizi_4d:
    '''

    书中例子，
    '''
    def __init__(self, bounds=None):
        '''
        :param bounds: 函数各维度的取值区间，用二维数组表示
        '''

        self.input_dim = 3
        if bounds is None:
            self.bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]
        else:
            self.bounds = bounds
        self.min = [0, 0, 0, 0]
        self.fmin = 0
        self.name = 'x2 ** 2 + x1 * x3 + x1 + x4 - 4'

    def f(self, x):
        '''
        :param X: 一维数组，具体的函数采样点
        :return: 具体的函数值
        '''
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        return x2 ** 2 + x1 * x3 + x1 + x4 * x2 * x3 - 4


class lizi_4d_2:
    '''

    书中例子，
    '''
    def __init__(self, bounds=None):
        '''
        :param bounds: 函数各维度的取值区间，用二维数组表示
        '''

        self.input_dim = 3
        if bounds is None:
            self.bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]
        else:
            self.bounds = bounds
        self.min = [0, 0, 0, 0]
        self.fmin = 0
        self.name = 'x2 ** 2 + x1 * x3 + x1 + x4 * x2 - 4'

    def f(self, x):
        '''
        :param X: 一维数组，具体的函数采样点
        :return: 具体的函数值
        '''
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        return x2 ** 2 + x1 * x3 + x1 + x4 * x2 - 4