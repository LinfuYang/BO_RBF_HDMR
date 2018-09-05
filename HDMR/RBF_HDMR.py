import numpy as np
from numpy.linalg import inv
import warnings
import random
from decimal import getcontext, Decimal
warnings.filterwarnings('ignore')

class Rbf_Hdmr:

    def __init__(self, func_objective=None):
        if func_objective == None:
            print('参数传入错误')
        else:
            self.func_mode = func_objective
            self.X_round = func_objective.bounds

    # 初始值的选择，观点一，随机采点，观点二，使用平均函数值的点, 观点三，在取值区间的中心范围内找点
    def x0_fx0(self):
        # Step 1  定义x0, f0
        m = len(self.X_round)
        self.x0 = np.zeros((m))
        for i in range(m):
            '''
            #  方法一 初始点采用所有维度中心位置的近似点
            vari_range = self.X_round[i][1] - self.X_round[i][0]
            self.x0[i] = round(self.X_round[i][0] + vari_range / 2 + random.uniform(-vari_range * 0.01, vari_range * 0.01), 2)
            '''
            # 方法二，随机选择一个点作为centercut
            self.x0[i] = round(random.uniform(self.X_round[i][0], self.X_round[i][1]), 2)

            # 方法三 初始化多个函数取值点，然后选择靠近函数均值的所对应的取值点作为center cut



        self.f0 = self.func_mode.f(self.x0)

        return self.x0, self.f0

    def exchange_point_1D(self, round, k):
        '''
        :param round:  为一维数组，里面存放的是要替换的元素
        :param k:  要替换元素的位置
        :return: 所有替换后额二维数组
        '''
        # 采用round中的点替换x0 中对应位置的点
        m = len(round)

        x_exchange = []
        for index in range(m):
            temp_x = self.x0.copy()
            temp_x[k] = round[index]
            # print(temp_x)
            x_exchange.append(temp_x)
        return np.array(x_exchange)

    # 计算对应的一位函数精确值
    def fun_value_1D(self, x_arr):
        '''
        :param x_arr: [x0, ...xn]，对应维度的自变量已发生变化
        :return: 返回一个列表，存放函数值
        '''
        fx_1 = []
        for i in range(len(x_arr)):

            fx_1.append(self.func_mode.f(x_arr[i]) - self.f0)

        return np.array(fx_1)

    # 得到A矩阵
    def A_func_1D(self, xy):
        '''
        :param xy:
        :return:
        '''
        m = len(xy)
        A = np.zeros((m,  m))
        # print(m)
        for i in range(m):
            for j in range(m):
                distance = xy[i] - xy[j]            # 如果xy为一位数组，则为对应元素相减，如果xy为多维数组，则为向量中对应元素相减
                if len(distance) == 1:          # 一维情况
                    One = distance[0] ** 2
                    Two = np.log(np.sqrt(One))
                elif len(distance) > 1:         # 多维情况
                    # print('distance:', distance)
                    One = 0
                    for index in range(len(distance)):
                        One += distance[index] ** 2
                    Two = np.log(np.sqrt(One))

                if Two == -(np.inf):
                    Two = 0
                A[i, j] = One * Two
        return A

    # 得到系数矩阵
    def RBF_func_1D(self, xy_arr, f_xy):
        '''
        最终得到函数系数
        :param xy:  表示某一维度上函数的采点值
        :param fxy: 与样本点的对应的一维函数值
        :return:
        '''
        xy = np.array(xy_arr.copy())            # 判断xy数组是几维数组
        a = xy.shape
        fxy = list(f_xy.copy())         # 转化为列表，便于添加元素
        if len(a) == 1:          # 如果是一维数组，将其变为二维列数组 N * 1 形式
            xy = xy.reshape(1, -1).T

        A = self.A_func_1D(xy)          # rbf 中的矩阵A
        n = np.shape(A)[1]
        # 加上线性的 p
        p_1 = np.ones(n).reshape(1, -1).T
        p_2 = xy
        p = np.hstack((p_1, p_2))               # 按航合并，也就是行要相等
        m = p.shape[1]
        zeros = np.zeros((m, m))            # 全零矩阵
        B = np.vstack((p, zeros))
        xishu_arr = np.hstack((np.vstack((A, p.T)), B))
        fxy.extend(np.zeros(m))
        f_ar = np.array(fxy).reshape(1, -1).T
        values = inv(xishu_arr).dot(f_ar)
        # print(values)
        round_values = []
        for index in values:
            x = index[0]
            y = float(round(x, 6))
            if y == -0.0:
                y = 0
            round_values.append(y)
        # 系数
        return np.array(round_values)

    # 求解一维线性函数的近似值
    def linear_func(self, coefs, round_x, x_i):
        # print('linear')
        # print('coefs:', coefs)
        # print('round_x:', round_x)
        # 根据 round_x 和 coefs 来求解函数表达式
        m = len(round_x)
        n = len(coefs)
        beta = coefs[:m]
        alpha = coefs[m - n:]
        p_x = list([1.0])
        p_x.append(x_i)
        # x_arr = np.array(list([x_i] * m)).reshape(1, -1)
        # print(x_arr)

        p = np.array(p_x).reshape(1, -1).T
        f_value = alpha.dot(p)

        return np.array(f_value)

    # 求解一维非线性函数的近似值
    def non_linear_func(self, coefs, round_x, x_i):
        '''
        :param coefs:  beta + alpha
        :param round_x: 代表采点矩阵个数
        :param x_i: 未知量，
        :return:
        '''
        if type(x_i) is not list:
            x_i = list([x_i])
            # print('x_i:', x_i)

        m = len(round_x)          # 采样点个数
        n = len(coefs)          # beta + alpha 长度
        beta = coefs[:m]                # beta
        alpha = coefs[m - n:]           # alpha 多项式
        p_x = list([1.0])               # 多项式
        p_x.extend(x_i)

        ln = len(x_i)
        if ln == 1:
            x_arr = np.array(list([x_i] * m)).reshape(1, -1).T
            x_value = np.array(round_x).reshape(1, -1).T

        else:
            x_arr = np.array([x_i] * m)

            x_value = np.array(round_x)
        # print('x_arr:', x_arr)
        # print('x_value:', x_value)
        x_arr = x_arr - x_value

        f_One_x = np.zeros((m, 1))
        if ln == 1:
            for index in range(m):
                One = x_arr[index] ** 2
                Two = np.log(abs(x_arr[index]))
                if Two == -(np.inf):
                    Two = 0
                f_One_x[index] = One * Two
        else:
            for index in range(m):
                One = x_arr[index, 0] ** 2 + x_arr[index, 1] ** 2

                Two = np.log(np.sqrt(One))

                if Two == -(np.inf):
                    Two = 0
                f_One_x[index] = One * Two

        f_value = beta.dot(f_One_x) + alpha.dot(p_x)

        return np.array([round(f_value[0], 6)])

    # 在函数是非线性函数的基础上进行抽样根据函数是不是线性函数进行抽样
    def sample_point(self,  round_x, point_x):

        point = round(random.uniform(round_x[0], round_x[1]), 2)
        # print('point_arr:', point_arr)
        # print('point:', point)
        flag = True
        '''
        vari_range = (round_x[1] - round_x[0]) * 0.01
        while flag:
            flag_temp = -1
            for ran_index in point_x:
                if point >= ran_index - vari_range and point <= ran_index + vari_range:
                    flag_temp = 1
            # print('flage_temp:', flag_temp)
            if flag_temp == 1:
                point = round(random.uniform(round_x[0], round_x[1]), 2)
            else:
                flag = False
        '''
        while flag:
            if point in point_x:
                point = round(random.uniform(round_x[0], round_x[1]), 2)
            else:
                flag = False

        return point


    # 对函数所有维度进行抽样
    def sample_func_1D(self, point_x_begin, round_x, k):
        '''
        :param point_x_begin:  初始左右链各个采样点
        :param round_x: 某一维的取值区间，表示形式为列表
        :param k: 表示在x0 中第几位需要改变
        :return:
        '''
        point_x = point_x_begin.copy()          # 将数转化为列表，便于添加元素
        point_x.append(round(self.x0[k], 2))

        # print('point_x:', point_x)
        # print('point_x:', np.array(point_x))

        x_exchange = self.exchange_point_1D(point_x, k)  # 变为三维采样点， 数组,其他维度不改变，只改变当前维度
        # print('x_exchange:', x_exchange)

        fx_2 = self.fun_value_1D(x_exchange)            # 计算对应的一维函数的函数值(精确值) # 数组
        # print('fx_2:', fx_2)

        One = round((fx_2[-1] - fx_2[1]) / (point_x[-1] - point_x[1]), 4)           # 根据左右端点, 判断该函数是不是线性函数,
        Two = round((fx_2[1] - fx_2[0]) / (point_x[1] - point_x[0]), 4)
        # print('One', One)
        # print('Two', Two)

        if One == Two:          # 1D线性情况
            type_f = 'linear'

        else:
            type_f = 'nolinear'
            random_x = self.sample_point(round_x, point_x)             # 采样， 三个点来模拟非线性函数不科学，所以采集第四个点

            # print('type(point_x):', type(point_x))
            point_x.append(random_x)
            flag = True

            while flag:
                # 每找一个新的采样点我们要重新构建近似函数，来查看函数的精确度。

                x_exchange = self.exchange_point_1D(point_x, k)         # 首先要计算采样点出的一维函数精确值
                fx_2 = self.fun_value_1D(x_exchange)
                # print('fx_2:', fx_2)

                index_arr = self.RBF_func_1D(point_x, fx_2)         # 函数系数，用于构建近似函数
                # print('index_arr:', index_arr)
                '''
                    由于随机选择一个点作为测试点来测试整个函数的精确度，是不准确的，所以我们选择两个点，
                    当两个点的精确度同时满足要求是，这是我们认为整个近似函数的精确度已达到要求
                '''

                # 测试点1，测试一阶函数和原函数（一阶）的近似程度 # 由一维变三维
                random_test_1 = self.sample_point(round_x, point_x)
                x_test_1 = self.exchange_point_1D(list([random_test_1]), k)
                # print('x_test:', x_test)
                tmp_jinque_1 = self.fun_value_1D(x_test_1)
                f_test_1 = round(tmp_jinque_1[0], 6)

                tmp_jinsi_1 = self.non_linear_func(index_arr, point_x, random_test_1)
                f_value_1 = round(tmp_jinsi_1[0], 6)
                # print('---1---:', f_test_1)
                # print('---1---:', f_value_1)
                #计算相对误差
                corr_err_1 = abs((f_test_1 - f_value_1) / f_test_1) * 100
                # print(corr_err)

                # 测试点2，测试一阶函数和原函数（一阶）的近似程度 # 由一维变三维
                random_test_2 = self.sample_point(round_x, point_x)
                x_test_2 = self.exchange_point_1D(list([random_test_2]), k)

                # print('x_test:', x_test)
                tmp_jinque_2 = self.fun_value_1D(x_test_2)
                f_test_2 = round(tmp_jinque_2[0], 6)
                tmp_jinsi_2 = self.non_linear_func(index_arr, point_x, random_test_2)
                f_value_2 = round(tmp_jinsi_2[0], 6)
                # print('----2---:', f_test_2)
                # print('----2---:', f_value_2)
                #计算相对误差
                corr_err_2 = abs((f_test_2 - f_value_2) / f_test_2) * 100


                # 偶尔放弃一个测试点
                if corr_err_1 <= 0.01 and corr_err_2 <= 0.01:
                    # print('random_test_1:', random_test_1, end=' ')
                    # print('random_test_2:', random_test_2)
                    # print('corr_err_1:', corr_err_1, end=' ')
                    # print('corr_err_2:', corr_err_2)
                    flag = False


                else:
                    random_x = self.sample_point(round_x, point_x)
                    # print('type(point_x):', type(point_x))
                    point_x.append(random_x)

        return type_f, np.array(point_x), np.array(fx_2)

    # 计算一阶函数模型近似值
    def func_DEMO(self):

        m = len(self.X_round)       # 计算原函数的维度
        self.point_round = []        # 保存每个维度的采样点
        self.f_values = []            # 保存每个一维函数对应采样点的函数值
        self.type_fx = []           # 保存每个维度的一维函数的类型
        self.xishu_arr = []         # 保存系数矩阵

        # 一种特殊情况
        if self.X_round[0] != self.X_round[1]:
            print('函数的自变量取值区间不同---------------------------------------------------------------------结束游戏')

        round_x = self.X_round[0]            # 这样写的前提是，每个自变量的取值区间是是一样的，要不然是要加循环的

        # 在自变量的取值区间的两个端点的邻近区间进行取值，来代表端点。由于邻近区间的长度为（x - 0.01 * len, x + 0.01 *len)， 这说明，在采样时只要保存小数点后两位小数就可以了
        variable_range = abs(round_x[-1] - round_x[0])
        left_x = round_x[0] + variable_range * 0.01      # 最小值
        right_x = round_x[-1] - variable_range * 0.01     # 最大值

        # left  保证采点与center cut 不同
        flag_left =True
        left_point = -np.inf
        while flag_left:
            left_point = round(random.uniform(round_x[0], left_x), 2)
            if left_point in self.x0:
                left_point = round(random.uniform(round_x[0], left_x), 2)
            else:
                flag_left =False

        # right 保证采点与center cut 不同
        flag_right = True
        right_point = -np.inf
        while flag_right:
            right_point = round(random.uniform(right_x, round_x[-1]), 2)
            if right_point in self.x0:
                right_point = round(random.uniform(right_x, round_x[-1]), 2)
            else:
                flag_right = False

        point_x_begin = []          #  新的采样端点
        point_x_begin.append(left_point)
        point_x_begin.append(right_point)

        for i in range(m):
            # print('******************************************************************************', i)
            type_f, list_x, f_value = self.sample_func_1D(point_x_begin, round_x, i)
            # print('type_f:', type_f)
            # print('list_x:', list_x)
            # print('f_value:', f_value)
            if type_f == 'linear':
                # 计算 系数矩阵
                index_arr = self.RBF_func_1D(list_x, f_value)
            else:
                index_arr = self.RBF_func_1D(list_x, f_value)

            # 保存每个维度的采样点
            self.point_round.append(list_x)
            # 保存每个维度的一维函数的类型
            self.type_fx.append(type_f)
            # 保存每个一维函数对应采样点的函数值
            self.f_values.append(f_value)
            # 保存系数矩阵
            self.xishu_arr.append(index_arr)
        return np.array(self.point_round), np.array(self.f_values), np.array(self.type_fx), np.array(self.xishu_arr)

    #  定义一个一阶函数，用于外界调用，算函数值
    def func_1D_value(self, xi, type=None, xishu=None, point_sample=None):
        '''
        :param type: 函数类型
        :param xishu: 系数矩阵
        :param point_sample: 构建模型采样的点
        :param xi: 自变量
        :return: 自变量的函数值
        '''

        if type == 'linear':
            f = self.linear_func(xishu, point_sample, xi)
        else:
            f = self.non_linear_func(xishu, point_sample, xi)
        return f

    def f_1d(self, x_i, k):
        '''
        :param x_i: 某个维度的自变量
        :param k:  该自变量是第几维
        :return:  近似函数值
        '''
        coefs = self.xishu_arr[k]
        xround = self.point_round[k]
        type_f = self.type_fx[k]
        # print(coefs)
        # print(xround)
        # print(type_f)

        if type_f == 'linear':
            f = self.linear_func(coefs, xround, x_i)
        else:
            f = self.non_linear_func(coefs, xround, x_i)

        return f

    # 原函数的一阶近似模型
    def func_1D(self, x_arr):

        f_value = self.f0
        for index in range(len(x_arr)):

            lala= self.f_1d(x_arr[index], index)
            # print('f1:', lala)
            f_value += lala
        return f_value

    def simulation_func(self):
        # 计算一下是否需要二阶的分解, 随便选取一个点，来计算精确值和近似值之间的差距
        flag = True
        random_x = np.zeros(len(self.x0))
        for i in range(len(self.x0)):

            random_x[i] = round(random.uniform(self.X_round[i][0], self.X_round[i][1]), 2)

        f_jinque = round(self.func_mode.f(random_x), 6)   # 精确值

        trmp = self.func_1D(random_x)    # 近似值
        f_jinsi = round(trmp[0], 6)

        # 如果相对误差大于0.01，说明一阶近似不能达到有效精度，我们要进行二阶有效计算
        corr_err = abs((f_jinque - f_jinsi) / f_jinque) * 100
        print('测试一阶函数模型精度')
        print('random_x:', random_x)
        print('f_jinque:', f_jinque)
        print('f_jinsi:', f_jinsi)
        print('corr_err:', corr_err)
        if corr_err > 0.1:
            flag = False
        return flag

    #  将二维数组变三维(xi,xj) -- (xi,xj,x0)
    def exchange_point_2D(self, round_ij, k):
        '''
        :param round:二维数组 [1, 0.18]
        :param k: 一维数组形式，存放我们要替换的两个点 [0, 1]
        :return: 三维【1， 0.18， 0.5】
        '''
        m = len(round_ij[0])
        n = len(round_ij[1])

        # 返回二维的采样点组合
        i_j = []

        x_exchange = []
        first = k[0]
        second = k[1]
        for i in range(m):
            for j in range(n):
                temp_x = np.copy(self.x0)
                temp_x[first] = round_ij[0][i]
                temp_x[second] = round_ij[1][j]
                x_exchange.append(temp_x)
                i_j.append(list([round_ij[0][i], round_ij[1][j]]))
        return x_exchange, i_j

    # 将一维xi,xj 变为（xi, xj）,
    def xixj_xij(self, round_x):
        '''
        :param round_x: 构建一阶函数模型时，每个维度的自变量的采样点
        :return:
        '''

        resulti_j_arr = []       # 返回二维的采样点组合
        result_point = []         #返回三维的采样点组合
        result_ij = []
        m = len(round_x)
        for i in range(m):
            for j in range(m):
                if i < j:
                    round_i_j = [round_x[i], round_x[j]]
                    point_k = list([i, j])
                    x_exchange, ij = self.exchange_point_2D(round_i_j, point_k)
                    result_point.append(x_exchange)
                    result_ij.append(point_k)
                    resulti_j_arr.append(ij)
        return result_point, result_ij, resulti_j_arr,

    # 要判断两个变量是不是相关，就需要从两个变量的样本点所组合成的三维函数样本点钟随机选取一个，带入一阶函数模型，如果函数通过该点，则说明两者无关。
    def is_correlation(self, inter_ij_1, inter_point_1):
        '''
        :param inter_ij_1: 自变量的两两排列组合
        :param inter_point_1: 与自变量两两排列组合相对应的自变量采样点。
        :return:
        '''
        random_point = []
        for i in range(len(inter_ij_1)):
            #　定义变量　保存每种组合的具体样本个数
            # 判断两个变量之间是否存在关系, 计算函数精确值和近似值，如果近似值与精确值差距很大，则说明二者无关.
            # 由于模型是用上述各维度的采样点所构建的，如果该新的组合点带入近似函数时，其近似值与精确值差别不大，则说明函数过该点，否则则说明，两者相关

            # 生成旧点
            k = inter_ij_1[i]
            # print('k;', k)
            # 使用两个一维自变量形成的高维样本点的组合
            # print('self.exchange_point_1D(self.point_round[k[0]], k[0]):', self.exchange_point_1D(self.point_round[k[0]], k[0]))
            # print('self.exchange_point_1D(self.point_round[k[1]], k[1]):', self.exchange_point_1D(self.point_round[k[1]], k[1]))

            x_i = np.vstack((self.exchange_point_1D(self.point_round[k[0]], k[0]), self.exchange_point_1D(self.point_round[k[1]], k[1])))
            # print(x_i)
            #  使用两个排列组合的二维自变量，再与cut center结合形成的高维样本点
            x_j = np.array(inter_point_1[i])
            # 为找到一个新点
            new_point = []
            for roc in range(len(x_j)):
                x = list(x_j[roc])
                flage = False
                for loc in range(len(x_i)):
                    y = list(x_i[loc])
                    if x == y:
                        flage = True
                if flage == False:
                    new_point.append(x)
            # print(new_point)
            random_point.append(new_point[0])

        # 二维数组保存相关性变量
        inter_arr = []
        for i in range(len(inter_ij_1)):
            # 精确值
            f_jinque = round(self.func_mode.f(random_point[i]), 6)
            # 近似值
            f_jinsi = round(self.func_1D(random_point[i])[0], 6)
            # print('f_jinque:', f_jinque)
            # print('f_jinsi:', f_jinsi)
            if abs((f_jinsi - f_jinque) / f_jinque) * 100 > 0.1:
                inter_arr.append(inter_ij_1[i])
        # print(inter_arr)
        return inter_arr


    def func_value2D(self, x_arr, corr_ij, point_round, f_values):
        '''
        :param x_arr: 二维数组，表示某两个自变量采样点组合后形成的三维采样点
        :param corr_ij: 哪两个变量进行组合的
        :return:
        '''

        fx_1 = []
        # 哪两个变量具有相关性
        left = corr_ij[0]
        right = corr_ij[1]

        # 表示一维函数的采样点
        round_1 = point_round[left]
        round_2 = point_round[right]

        #  一维采样点的函数值
        fx1 = f_values[left]
        fx2 = f_values[right]

        for i in range(len(x_arr)):
            # 表示函数在两个维度的一阶函数近似值
            value = x_arr[i]
            # 在三维数组的某一个值里面寻找对应的自变量的气质取值
            # 根据下标去找函数值
            f_1 = fx1[list(round_1).index(value[left])]
            f_2 = fx2[list(round_2).index(value[right])]

            fx_1.append(self.func_mode.f(x_arr[i]) - self.f0 - f_1 - f_2)

        return fx_1

    def func_2(self):
        '''
        :return:求解函数的参数
        '''
        # 根据新产生的点，来判断两个自变量之间是否存在关系, 自变量之间存在关系的有哪些，分别是哪两个之间的关系
        inter_arr = self.is_correlation(self.result_ij, self.result_point)
        x_ij_index = inter_arr         # 相关变量的二维变量组合的下标表示
        x_ij_point = []                # 相关变量的二维变量组合
        x_ij_value = []                # 相关变量的二维变量组合的函数值
        x_ij_xishu = []                # 相关变量的二维变量组合函数的系数矩阵
        if len(inter_arr) != 0:
            for i in range(len(inter_arr)):
                # 指针，指向inter_arr[i] 在排列着中的位置
                index_i = self.result_ij.index(inter_arr[i])         # [[0, 2]]

                # 三维采样点
                point_ij = self.result_point[index_i]                  # 2d -3d

                func_2D = self.func_value2D(point_ij, inter_arr[i], self.point_round, self.f_values)    # 计算函数值

                sample_ij = self.resulti_j_arr[index_i]                  #  求函数值所需要的两个变量的排列组合的点

                xishu_2D = self.RBF_func_1D(sample_ij, func_2D)                  # 系数矩阵

                x_ij_point.append(sample_ij)

                x_ij_value.append(func_2D)

                x_ij_xishu.append(xishu_2D)

        return np.array(x_ij_index), np.array(x_ij_point), np.array(x_ij_value), np.array(x_ij_xishu)

    # 求每个二阶模型的近似值

    # 求二阶模型函数近似值
    def f_2d(self, x_arr, index_ij=None, xishu=None, point_sample=None):
        '''
        :param x_arr: 自变量
        :param index_ij: 两个相关变量的下标
        :param xishu: 相关变量的系数矩阵
        :param point_sample: 两个相关变量在构建模型时的采样点
        :return: 二阶函数的函数值
        '''
        f2 = 0
        for i in range(len(self.x_ij_index)):

            indx_1 = index_ij[i][0]
            indx_2 = index_ij[i][1]
            x12 = list([x_arr[indx_1], x_arr[indx_2]])
            # print('f2:', f2)
            f_tmp = self.non_linear_func(xishu[i], point_sample[i], x12)
            # print(f_tmp)
            f2 += f_tmp
        return f2

    # 计算任意两个相关变量之间的近似值
    def func_2D_value(self, x12, index_ij=None, xishu=None, points=None):
        f2 = 0
        # print('x12:', x12)
        f_tmp = self.non_linear_func(xishu, points, x12)
        # print(f_tmp)
        f2 += f_tmp
        return f2

    # 查看一阶函数模型是不是精确
    def f_2DM(self, x_arr):
        f_m = 0
        isJingque = self.simulation_func()
        f_1 = self.func_1D(x_arr)
        if isJingque == True:
            f_m = f_1
        else:
            # 2D
            # result_point 代表的是关于一维xi和xj　生成的三维样本点的排列组合
            # result_ij 代表三维排列组合分别是有哪两个变量组合而来的
            # resulrij_arr 代表有两个一维的xi 和xj 的二维排列组合，
            self.result_point, self.result_ij, self.resulti_j_arr = self.xixj_xij(self.point_round)

            # x_ij_index 相关变量的二维变量组合的下标表示
            # x_ij_point 相关变量的二维变量组合
            # x_ij_value 相关变量的二维变量组合的函数值
            # x_ij_xishu 相关变量的二维变量组合函数的系数矩阵
            # print('1234568971111111111111111', f_1)
            self.x_ij_index, self.x_ij_point, self.x_ij_value, self.x_ij_xishu = self.func_2()

            f_2 = self.f_2d(x_arr, index_ij=self.x_ij_index, xishu=self.x_ij_xishu, point_sample=self.x_ij_point)
            # print('f_2:', f_2)
            f_m = f_1 + f_2
        return f_m

    def f_Two_index(self):

        # 2D
        # result_point 代表的是关于一维xi和xj　生成的三维样本点的排列组合
        # result_ij 代表三维排列组合分别是有哪两个变量组合而来的
        # resulrij_arr 代表有两个一维的xi 和xj 的二维排列组合，
        self.result_point, self.result_ij, self.resulti_j_arr = self.xixj_xij(self.point_round)

        # x_ij_index 相关变量的二维变量组合的下标表示
        # x_ij_point 相关变量的二维变量组合
        # x_ij_value 相关变量的二维变量组合的函数值
        # x_ij_xishu 相关变量的二维变量组合函数的系数矩阵
        # print('1234568971111111111111111', f_1)
        self.x_ij_index, self.x_ij_point, self.x_ij_value, self.x_ij_xishu = self.func_2()

        return self.x_ij_index, self.x_ij_point, self.x_ij_value, self.x_ij_xishu
