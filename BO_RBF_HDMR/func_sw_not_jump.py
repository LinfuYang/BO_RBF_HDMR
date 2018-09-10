from HDMR import RBF_HDMR
from Objective_function import Func_Nd
from collections import OrderedDict
import numpy as np
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import matern52
import random
import matplotlib.pyplot as plt
import pandas as pd
import warnings

f_objective = Func_Nd.Schwefel_Func(input_dim=5)
rbf_hdmr = RBF_HDMR.Rbf_Hdmr(f_objective)

# 其变量取值区间
x_round = f_objective.bounds

# 计算x0,f0。第一种选择方法，随机选取
x0, f0 = rbf_hdmr.x0_fx0()
print('center cut point:', x0)
print('center cut function', f0)

point_round, f_values, type_fx, xishu_arr = rbf_hdmr.func_DEMO()

print('**************************一阶函数模型参数*******************************')

print('采集点', point_round)
# print('采集点的函数值', f_values)
print('type_fx', type_fx)
# print('系数矩阵', xishu_arr)
# 查看一阶模型的精度情况
is_True = rbf_hdmr.simulation_func()
print('is_True:', is_True)

if is_True == False:
    # 由于我们规定，任何一种变量只能与其他某一种变量发生一次相关系，即 没一个变量的下标在x_ij_index数组中最多可以出现一次
    x_ij_index, x_ij_point, x_ij_value, x_ij_xishu = rbf_hdmr.f_Two_index()

    print('x_ij_index:', x_ij_index)
    # print(x_ij_point)
    # print(x_ij_value)
    # print(x_ij_xishu)

    # 先判断哪些自变量是与其他自变量无关的
    # 整合相关自变量的数组
    if len(x_ij_index) != 0:
        x_inter = []
        for index in range(len(x_ij_index)):
            x_inter.append(list(x_ij_index[index]))
        x_inter = np.unique(x_inter)
        # print(x_inter)
        # print(x_ij_index)
        # print(x_ij_point)
        # print(x_ij_value)
        # print(x_ij_xishu)
    print('*********************************函数模型搭建完毕************************************')


def func_first_order(type_f=None, xishu_f=None, point_f=None):
    # 独立变量的线性、非线性判断
    if type_f == 'linear':
        print('执行一阶线性优化')
        # 左端点函数值
        f_left = rbf_hdmr.func_1D_value(x_round[0][0], type=type_f, xishu=xishu_f,
                                        point_sample=point_f)
        # 右端点函数值
        f_right = rbf_hdmr.func_1D_value(x_round[0][1], type=type_f, xishu=xishu_f,
                                         point_sample=point_f)
        if f_left > f_right:
            f_min_i = f_right
            x_min = x_round[0][1]
        else:
            f_min_i = f_left
            x_min = x_round[0][0]

    # 独立变量的非线性情况
    else:
        print('执行一阶非线性函数优化')

        # 非一维线性函数最好的办法采用BO来找函数最小值
        def f(x):
            return -(rbf_hdmr.func_1D_value(x, type=type_f, xishu=xishu_f, point_sample=point_f))

        sexp = matern52()
        gp = GaussianProcess(sexp)
        acq = Acquisition(mode='ExpectedImprovement')
        round_x = (x_round[0][0], x_round[0][1])
        param = {'x': ('cont', round_x)}
        gpgo = GPGO(gp, acq, f, param)
        gpgo.run(max_iter=20, nstart=10)
        res, f_min_i = gpgo.getResult()

        print('res:', res)
        x_min = res[0]

    return x_min, f_min_i


def is_xiangguan_2D(second_order=None):
    # second_order用来存放变量存在共项元素的关系
    # 例如：denpend_func_2: [[0, 1, 2], [1, 2], [2]] 表示 second_order中第一项与第二项、第三项共享元素，第一项和第二项共享元素
    xishu = []
    i = 0
    # print('the type of second_order:',type(second_order))
    while i < len(second_order):
        index_x = []
        for inx in xishu:
            index_x.extend(inx)
        index_x = list(np.unique(index_x))
        # print('index_x:', index_x)
        while i < len(index_x):
            if i in index_x:
                i += 1
            else:
                break

        if i == len(second_order):
            break
        else:
            # print(i)
            temp = list(second_order[i].copy())
            index = [i]
            j = 0
            while j < len(second_order):
                if j not in index:
                    if (second_order[j][0] in temp) or (second_order[j][1] in temp):
                        index.append(j)
                        temp.extend(list(second_order[j]))
                        temp = list(np.unique(temp))
                        index = list(np.unique(index))
                        j = 0
                    else:
                        j += 1
                else:
                    j = j + 1
            # print(temp)
            # print(index)
            xishu.append(index)
            i += 1

    # 判断一维和二维
    '''
    由于denpend_d表示的是二维函数间是否存在共享变量
    '''
    depend_2D = []
    independ_2D = []
    # 不存一阶函数的情况
    for temp in xishu:
        if len(temp) > 1:
            depend_2D.append(temp)
        else:
            independ_2D.append(temp)

    return depend_2D, independ_2D


def func_model(index_ij=None, x_min=None, func_min=None, max_iter_i=10, nstart_i=10):
    '''
    :param index_ij: 选择的函数项
    :param init_x: 初始的函数最优解
    :return: 本地迭代函数最优解
    '''
    first_order = []
    second_order = []
    for k in index_ij:
        if k < len(type_fx):
            # 一阶
            first_order.append(k)
        else:
            second_order.append(x_ij_index[k - len(type_fx)])

    # 二阶中的变量
    x_inter = []
    if len(second_order) != 0:

        for index in range(len(second_order)):
            x_inter.append(list(second_order[index]))
        x_inter = np.unique(x_inter)
    print('first_order:', first_order)
    print('second_order:', second_order)
    print('x_inter:', x_inter)

    # 定义优化维度
    temp_first = first_order.copy()
    temp_first.extend(x_inter)
    index_dimen = np.unique(temp_first)

    # 定义一个数组，用来存放需要和二阶函数一起计算的一阶函数自变量代号
    # 如果所有的一阶函数和二阶函数无关的话，该数组为空，且下面代码自动求取最小值，并根据坐标添加到对应的自变量取值和函数最小值中
    denpend_point_1D = []
    if len(first_order) != 0:
        for i in first_order:
            # print(i)
            # print('x_inter:', x_inter)
            # 独立情况
            if i not in x_inter:
                type_fx_i = type_fx[i]
                xishu_arr_i = xishu_arr[i]
                point_round_i = point_round[i]
                print('执行一阶不相关变量的优化, 函数下标为:', i)
                min_x, min_f = func_first_order(type_f=type_fx_i, xishu_f=xishu_arr_i, point_f=point_round_i)
                # print('min_x:', min_x)
                x_min[i] = min_x
                # print('一阶线性无关', x_min)
                func_min += min_f
            else:
                denpend_point_1D.append(i)

    '''
     这里只存在相关的变量 # 函数代号剩余存放在denpend_point(肯定和二维函数共项变量的一维函数) 和 second_order
    接下来要判断一阶函数和哪些二阶函数具有共同的自变量
        1.判断二阶函数中有哪些共项
    '''
    # print('denpend_point_1D:', denpend_point_1D)
    # print('second_order:', second_order)
    # 解决二维情况，分解为二维相关和二维无关
    X = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
         'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # 判断二阶变量是否具有相关性
    depend_2D, independ_2D = is_xiangguan_2D(second_order=second_order)
    print('depend_2D:', depend_2D)
    print('independ_2D:', independ_2D)
    # 要判断一维和二维的关系

    # 存在一阶与二阶相关的函数，即二阶函数不为空
    if len(denpend_point_1D) != 0:

        # 情况分为两种：一维函数与二维相关函数有关
        #             一维函数与二维不相关函数有关
        # 1. 构建一个二维数组，行表示一阶非独立函数，列表示二阶非独立函数或者二阶独立函数
        # denpend_point_1D + depend_2D
        if len(depend_2D) != 0:

            #  二维相关函数变量
            unique_2D = []
            for i in range(len(depend_2D)):
                temp_0i = []
                for j in range(len(depend_2D[i])):
                    aaa = list(second_order[depend_2D[i][j]])
                    temp_0i.append(list(aaa))
                temp_0i = np.unique(temp_0i)
                unique_2D.append(temp_0i)
            # print('unique_2d:----------------', unique_2D)
            # 求解depend_2D数组中所对应的second——order中数组去掉相同项
            # 查看每一列，找到数值为1的项，说明该一维函数和对应的二维函数相关。
            flag_arr = np.array([[-1] * len(unique_2D)] * len(denpend_point_1D))

            for row in range(len(denpend_point_1D)):
                for col in range(len(unique_2D)):
                    if denpend_point_1D[row] in list(unique_2D[col]):
                        flag_arr[row, col] = 1

            # print('flag_arr:', flag_arr)
            # 定义二维数组，存放1D函数 和2D函数的的关系,里面的每个一维数组的第一个元素代表二维函数在depend_2D中的下标，后面的每个元素都代表一个一维函数
            f1_f2 = []
            # 先访问列
            for col in range(len(depend_2D)):
                # 再访问行
                f2D = [col]
                for row in range(len(denpend_point_1D)):
                    if flag_arr[row, col] == 1:
                        f2D.append(row)
                f1_f2.append(f2D)
            # print('f1_f2:', f1_f2)
            for row in range(len(f1_f2)):
                # 表示一阶函数与二阶函数存在相关变量
                if len(f1_f2[row]) > 1:
                    print('执行一阶函数与二阶相关函数的优化')
                    # # 一维函数好多,里边每个元素的标号代表函数编号，而且也是自变量编号
                    f_1_depend = f1_f2[row][1:]
                    # print('f_1_depend:', f_1_depend)
                    # 为了找到一维函数和二维函数一共使用了多少变量， #将一维和二维的自变量并起来，并去重

                    # 构建二维函数的系数矩阵
                    ij_index_i = []
                    ij_xishu_i = []
                    ij_point_i = []
                    for i in range(len(depend_2D[row])):
                        ij = second_order[depend_2D[row][i]]
                        # print('ij:', ij)
                        # 为了找函数系数
                        index = -1
                        for j in range(len(x_ij_index)):
                            if x_ij_index[j][0] == ij[0] and x_ij_index[j][1] == ij[1]:
                                index = j
                            # print('index:', index)
                        ij_index_i.append(x_ij_index[index])
                        ij_xishu_i.append(x_ij_xishu[index])
                        ij_point_i.append(x_ij_point[index])

                    # print('ij_index_i:', ij_index_i)
                    len_ij = np.unique(np.array(ij_index_i))

                    # 数组，存放使用变量的情况，里面编号，表示是那个维度的的变量
                    X_name = []
                    for x in range(len(len_ij)):
                        X_name.append(X[len_ij[x]])

                    def f(X_name):
                        f_index = 0
                        # 一阶函数
                        for i in range(len(f_1_depend)):
                            type_fx_1 = type_fx[f_1_depend[i]]
                            xishu_arr_1 = xishu_arr[f_1_depend[i]]
                            point_round_1 = point_round[f_1_depend[i]]
                            point_index = -1
                            for x in range(len(len_ij)):
                                if f_1_depend[i] == len_ij[x]:
                                    point_index = x
                            x_name = X_name[point_index]
                            f_index += -(rbf_hdmr.func_1D_value(x_name, type=type_fx_1, xishu=xishu_arr_1,
                                                                point_sample=point_round_1))

                        for index in range(len(depend_2D[row])):
                            ij_index = ij_index_i[index]
                            ij_xishu = ij_xishu_i[index]
                            ij_point = ij_point_i[index]
                            # print('ij_index:', ij_index)
                            # print('X_name:', X_name)
                            left = -1
                            right = -1
                            for x in range(len(len_ij)):
                                if ij_index[0] == len_ij[x]:
                                    left = x
                                if ij_index[1] == len_ij[x]:
                                    right = x
                            x_name = [X_name[left], X_name[right]]
                            # print('x_name:', x_name)
                            f_index += -(
                                rbf_hdmr.func_2D_value(x_name, index_ij=ij_index, xishu=ij_xishu, points=ij_point))
                        return f_index

                    param = OrderedDict()
                    for m in range(len(len_ij)):
                        # print('x_round[ij_index_i[i]]', x_round[len_ij[m]])
                        param[X_name[m]] = ('cont', x_round[len_ij[m]])
                    # print(param)
                    sexp = matern52()
                    gp = GaussianProcess(sexp)
                    acq = Acquisition(mode='ExpectedImprovement')
                    gpgo = GPGO(gp, acq, f, param)
                    gpgo.run(max_iter=max_iter_i, nstart=nstart_i)
                    res, max_xy = gpgo.getResult()
                    # print('ij:', ij)
                    for x in range(len(len_ij)):
                        x_min[len_ij[x]] = res[x]
                    # print('x_min:', x_min)
                    func_min += max_xy

                # 说明该一阶函数二阶相关函数独立
                else:
                    print('一阶函数不与二阶不相关函数相关')
                    # f_2_x表示在depend_2D中对应的相关项
                    f_2_x = depend_2D[f1_f2[row][0]]
                    # print('f_2_x:', f_2_x)
                    # 构建二维函数的系数矩阵
                    ij_index_i = []
                    ij_xishu_i = []
                    ij_point_i = []
                    for i in range(len(depend_2D[row])):
                        ij = second_order[depend_2D[row][i]]
                        # print('ij:', ij)
                        # 为了找函数系数
                        index = -1
                        for j in range(len(x_ij_index)):
                            if x_ij_index[j][0] == ij[0] and x_ij_index[j][1] == ij[1]:
                                index = j
                            # print('index:', index)
                        ij_index_i.append(x_ij_index[index])
                        ij_xishu_i.append(x_ij_xishu[index])
                        ij_point_i.append(x_ij_point[index])
                    # print('ij_index_i:', ij_index_i)
                    len_ij = np.unique(np.array(ij_index_i))

                    # 数组，存放使用变量的情况，里面编号，表示是那个维度的的变量
                    X_name = []
                    for x in range(len(len_ij)):
                        X_name.append(X[len_ij[x]])

                    def f(X_name):

                        f_index = 0
                        for index in range(len(depend_2D[row])):
                            ij_index = ij_index_i[index]
                            ij_xishu = ij_xishu_i[index]
                            ij_point = ij_point_i[index]
                            # print('ij_index:', ij_index)
                            # print('X_name:', X_name)
                            left = -1
                            right = -1
                            for x in range(len(len_ij)):
                                if ij_index[0] == len_ij[x]:
                                    left = x
                                if ij_index[1] == len_ij[x]:
                                    right = x
                            #  二阶函数
                            x_name = [X_name[left], X_name[right]]
                            # print('x_name:', x_name)
                            f_index += -(
                                rbf_hdmr.func_2D_value(x_name, index_ij=ij_index, xishu=ij_xishu, points=ij_point))
                        return f_index

                    param = OrderedDict()
                    for m in range(len(len_ij)):
                        # print('x_round[ij_index_i[i]]', x_round[len_ij[m]])
                        param[X_name[m]] = ('cont', x_round[len_ij[m]])
                    # print(param)
                    sexp = matern52()
                    gp = GaussianProcess(sexp)
                    acq = Acquisition(mode='ExpectedImprovement')
                    gpgo = GPGO(gp, acq, f, param)
                    gpgo.run(max_iter=max_iter_i, nstart=nstart_i)
                    res, max_xy = gpgo.getResult()
                    # print('ij:', ij)
                    for x in range(len(len_ij)):
                        x_min[len_ij[x]] = res[x]
                    # print('x_min:', x_min)
                    func_min += max_xy

        if len(independ_2D) != 0:
            #  二维不相关函数变量
            unique_2D = []
            for i in range(len(independ_2D)):
                temp_0i = []
                for j in range(len(independ_2D[i])):
                    aaa = list(second_order[independ_2D[i][j]])
                    temp_0i.append(list(aaa))
                temp_0i = np.unique(temp_0i)
                unique_2D.append(temp_0i)
            # print('unique_2d:----------------', unique_2D)

            # 求解independ_2D数组中所对应的second——order中数组去掉相同项
            # 查看每一列，找到数值为1的项，说明该一维函数和对应的二维函数相关。
            flag_arr = np.array([[-1] * len(unique_2D)] * len(denpend_point_1D))
            # print(flag_arr)
            for row in range(len(denpend_point_1D)):
                for col in range(len(unique_2D)):
                    if denpend_point_1D[row] in list(unique_2D[col]):
                        flag_arr[row, col] = 1
            # print(flag_arr)
            # 定义二维数组，存放1D函数 和2D函数的的关系,里面的每个一维数组的第一个元素代表二维函数在depend_2D中的下标，后面的每个元素都代表一个一维函数
            f1_f2 = []
            # 先访问列
            for col in range(len(independ_2D)):
                # 再访问行
                f2D = [col]
                for row in range(len(denpend_point_1D)):
                    if flag_arr[row, col] == 1:
                        f2D.append(row)
                f1_f2.append(f2D)
            # print('f1_f2:', f1_f2)
            for row in range(len(f1_f2)):
                # 表示一阶函数与二阶函数存在相关变量
                if len(f1_f2[row]) > 1:
                    print('执行一阶函数与二阶非相关函数的优化')
                    # f_2_x表示在independ_2D中对应的相关项
                    f_2_x = independ_2D[f1_f2[row][0]]
                    # print('f_2_x:', f_2_x)
                    # 一维函数好多,里边每个元素的标号代表函数编号，而且也是自变量编号
                    f_1_depend = f1_f2[row][1:]
                    # print('f_1_depend:', f_1_depend)
                    # 为了找到一维函数和二维函数一共使用了多少变量， #将一维和二维的自变量并起来，并去重
                    # 一二维函数自变量的自变量(只要确定二维函数使用了哪些变量就可以)
                    f_1 = unique_2D[row]
                    # print('f_1:', f_1)

                    # 构建二维函数的系数矩阵
                    ij_index_i = []
                    ij_xishu_i = []
                    ij_point_i = []
                    for i in range(len(independ_2D[row])):
                        ij = second_order[independ_2D[row][i]]
                        # print('ij:', ij)
                        # 为了找函数系数
                        index = -1
                        for j in range(len(x_ij_index)):
                            if x_ij_index[j][0] == ij[0] and x_ij_index[j][1] == ij[1]:
                                index = j
                            # print('index:', index)
                        ij_index_i.append(x_ij_index[index])
                        ij_xishu_i.append(x_ij_xishu[index])
                        ij_point_i.append(x_ij_point[index])

                    # print('ij_index_i:', ij_index_i)
                    len_ij = np.unique(np.array(ij_index_i))

                    # 数组，存放使用变量的情况，里面编号，表示是那个维度的的变量
                    X_name = []
                    for x in range(len(len_ij)):
                        X_name.append(X[len_ij[x]])

                    def f(X_name):
                        f_index = 0
                        # 一阶函数
                        for i in range(len(f_1_depend)):
                            # print('f_1_depend:', f_1_depend)
                            type_fx_1 = type_fx[f_1_depend[i]]
                            xishu_arr_1 = xishu_arr[f_1_depend[i]]
                            point_round_1 = point_round[f_1_depend[i]]
                            point_index = -1
                            for x in range(len(len_ij)):
                                if f_1_depend[i] == len_ij[x]:
                                    point_index = x
                            x_name = X_name[point_index]
                            f_index += -(rbf_hdmr.func_1D_value(x_name, type=type_fx_1, xishu=xishu_arr_1,
                                                                point_sample=point_round_1))
                        for index in range(len(independ_2D[row])):
                            ij_index = ij_index_i[index]
                            ij_xishu = ij_xishu_i[index]
                            ij_point = ij_point_i[index]
                            # print('ij_index:', ij_index)
                            # print('X_name:', X_name)
                            left = -1
                            right = -1
                            for x in range(len(len_ij)):
                                if ij_index[0] == len_ij[x]:
                                    left = x
                                if ij_index[1] == len_ij[x]:
                                    right = x
                            x_name = [X_name[left], X_name[right]]
                            # print('x_name:', x_name)
                            f_index += -(
                                rbf_hdmr.func_2D_value(x_name, index_ij=ij_index, xishu=ij_xishu, points=ij_point))
                        return f_index

                    param = OrderedDict()
                    for m in range(len(len_ij)):
                        # print('x_round[ij_index_i[i]]', x_round[len_ij[m]])
                        param[X_name[m]] = ('cont', x_round[len_ij[m]])
                    # print(param)
                    sexp = matern52()
                    gp = GaussianProcess(sexp)
                    acq = Acquisition(mode='ExpectedImprovement')
                    gpgo = GPGO(gp, acq, f, param)
                    gpgo.run(max_iter=max_iter_i, nstart=nstart_i)
                    res, max_xy = gpgo.getResult()
                    # print('ij:', ij)
                    for x in range(len(len_ij)):
                        x_min[len_ij[x]] = res[x]

                    func_min += max_xy
                    # print('x_min:', x_min)
                    # 说明该一阶函数二阶相关函数独立
                else:
                    print('执行二阶非相关函数的优化')
                    # print('duli')
                    # f_2_x表示在depend_2D中对应的相关项
                    f_2_x = independ_2D[f1_f2[row][0]]
                    # print('f_2_x:', f_2_x)
                    # 构建二维函数的系数矩阵
                    ij_index_i = []
                    ij_xishu_i = []
                    ij_point_i = []
                    for i in range(len(independ_2D[row])):
                        ij = second_order[independ_2D[row][i]]
                        # print('ij:', ij)
                        # 为了找函数系数
                        index = -1
                        for j in range(len(x_ij_index)):
                            if x_ij_index[j][0] == ij[0] and x_ij_index[j][1] == ij[1]:
                                index = j
                            # print('index:', index)
                        ij_index_i.append(x_ij_index[index])
                        ij_xishu_i.append(x_ij_xishu[index])
                        ij_point_i.append(x_ij_point[index])
                    # print('ij_index_i:', ij_index_i)
                    len_ij = np.unique(np.array(ij_index_i))

                    # 数组，存放使用变量的情况，里面编号，表示是那个维度的的变量
                    X_name = []
                    for x in range(len(len_ij)):
                        X_name.append(X[len_ij[x]])

                    def f(X_name):
                        f_index = 0
                        #  二阶函数
                        for index in range(len(independ_2D[row])):
                            ij_index = ij_index_i[index]
                            ij_xishu = ij_xishu_i[index]
                            ij_point = ij_point_i[index]
                            # print('ij_index:', ij_index)
                            # print('X_name:', X_name)
                            left = -1
                            right = -1
                            for x in range(len(len_ij)):
                                if ij_index[0] == len_ij[x]:
                                    left = x
                                if ij_index[1] == len_ij[x]:
                                    right = x
                            x_name = [X_name[left], X_name[right]]
                            # print('x_name:', x_name)
                            f_index += -(
                                rbf_hdmr.func_2D_value(x_name, index_ij=ij_index, xishu=ij_xishu, points=ij_point))
                        return f_index

                    param = OrderedDict()
                    for m in range(len(len_ij)):
                        # print('x_round[ij_index_i[i]]', x_round[len_ij[m]])
                        param[X_name[m]] = ('cont', x_round[len_ij[m]])
                    # print(param)
                    sexp = matern52()
                    gp = GaussianProcess(sexp)
                    acq = Acquisition(mode='ExpectedImprovement')
                    gpgo = GPGO(gp, acq, f, param)
                    gpgo.run(max_iter=max_iter_i, nstart=nstart_i)
                    res, max_xy = gpgo.getResult()
                    # print('ij:', ij)
                    for x in range(len(len_ij)):
                        x_min[len_ij[x]] = res[x]
                    # print('x_min:', x_min)
                    func_min += max_xy

    # 只存在二维相关性问题
    elif len(denpend_point_1D) == 0:
        # 解决一维与二维不存在相关变量且二维非先关变量的函数
        if len(independ_2D) != 0:
            print('执行二阶不相关变量的优化')
            for i in range(len(independ_2D)):
                ij = second_order[independ_2D[i][0]]
                # 在相关数组中的坐标
                index = -1
                for j in range(len(x_ij_index)):
                    if x_ij_index[j][0] == ij[0] and x_ij_index[j][1] == ij[1]:
                        index = j
                # print(index)
                ij_index = x_ij_index[index]
                ij_xishu = x_ij_xishu[index]
                ij_point = x_ij_point[index]
                # print('ij_index:', ij_index)
                X_name = [X[ij_index[0]], X[ij_index[1]]]

                def f(X_name):
                    return -(rbf_hdmr.func_2D_value(X_name, index_ij=ij_index, xishu=ij_xishu, points=ij_point))

                param = OrderedDict()
                for m in range(len(ij_index)):
                    # print('132', x_round[ij_index[i]])
                    param[X_name[m]] = ('cont', x_round[ij_index[m]])
                # print(param)
                sexp = matern52()
                gp = GaussianProcess(sexp)
                acq = Acquisition(mode='ExpectedImprovement')
                gpgo = GPGO(gp, acq, f, param)
                gpgo.run(max_iter=max_iter_i, nstart=nstart_i)
                res, max_xy = gpgo.getResult()
                # print('ij:', ij)
                # print(res)
                for hiahia in range(len(ij_index)):
                    x_min[ij_index[hiahia]] = res[hiahia]

                func_min += max_xy
                # print('x_min:', x_min)
                # print('f_min:', func_min)
        # 解决二维相关变量问题
        if len(depend_2D) != 0:
            print('执行二阶相关变量的优化')
            for i in range(len(depend_2D)):
                temp = depend_2D[i]
                # print('temp:', temp)
                ij_index_i = []
                ij_xishu_i = []
                ij_point_i = []
                for k in range(len(temp)):
                    ij = second_order[temp[k]]
                    # 为了找函数系数
                    index = -1
                    for j in range(len(x_ij_index)):
                        if x_ij_index[j][0] == ij[0] and x_ij_index[j][1] == ij[1]:
                            index = j
                    ij_index_i.append(x_ij_index[index])
                    ij_xishu_i.append(x_ij_xishu[index])
                    ij_point_i.append(x_ij_point[index])

                len_ij = np.unique(np.array(ij_index_i))
                # 数组，存放使用变量的情况，里面编号，表示是那个维度的的变量
                X_name = []
                for x in range(len(len_ij)):
                    X_name.append(X[len_ij[x]])

                def f(X_name):

                    f_index = 0
                    for index in range(len(ij_index_i)):
                        ij_index = ij_index_i[index]
                        ij_xishu = ij_xishu_i[index]
                        ij_point = ij_point_i[index]
                        # print('ij_index:', ij_index)
                        # print('X_name:', X_name)
                        left = -1
                        right = -1
                        for x in range(len(len_ij)):
                            if ij_index[0] == len_ij[x]:
                                left = x
                            if ij_index[1] == len_ij[x]:
                                right = x
                        #  二阶函数
                        x_name = [X_name[left], X_name[right]]
                        # print('x_name:', x_name)
                        f_index += -(rbf_hdmr.func_2D_value(x_name, index_ij=ij_index, xishu=ij_xishu, points=ij_point))
                    return f_index

                param = OrderedDict()
                for m in range(len(len_ij)):
                    # print('x_round[ij_index_i[i]]', x_round[len_ij[m]])
                    param[X_name[m]] = ('cont', x_round[len_ij[m]])
                # print(param)
                sexp = matern52()
                gp = GaussianProcess(sexp)
                acq = Acquisition(mode='ExpectedImprovement')
                gpgo = GPGO(gp, acq, f, param)
                gpgo.run(max_iter=max_iter_i, nstart=nstart_i)
                res, max_xy = gpgo.getResult()
                # print('ij:', ij)
                for x in range(len(len_ij)):
                    x_min[len_ij[x]] = res[x]
                # print('x_min:', x_min)
                func_min += max_xy
    # 现在只剩下一维和二维相关的两种函数，但是并不知道谁和谁相关
    '''
        但我们知道的是贝叶斯优化在低维阶段有较高的优化能力(D<=5)，所以我的想法是
        1.采用精确函数，一阶、二阶精确函数进行计算（待实现）
        2.采用近似函数，一阶、二阶近似函数进行计算（本代码采用）
            分类三类函数：
                只存在一阶函数(已解决)
                只存在二阶函数(且不存在共项变量的情况已解决)
                即存在一阶函数，还存在二阶函数
    '''

    return x_min, func_min, index_dimen


def f_bo(num_iter=100):
    num = num_iter
    # a 表示的是所有函数累加起来的长度，a -1 表示函数下标
    a = -1
    if is_True == True or len(x_ij_index) == 0:
        a = len(type_fx) - 1
    else:
        a = len(type_fx) + len(x_ij_index) - 1
    print('最后一个子函数的下标是：', a, )

    x_ending = x0
    f_ending = f0

    f_list = []

    for i in range(num):
        print('**********************************第', i, '次迭代**********************************')
        random_arr = []
        random_k = 5
        for j in range(random_k):
            temp = random.randint(0, a)
            flag = True
            while flag:

                if temp in random_arr:
                    temp = random.randint(0, a)
                else:
                    flag = False
                    random_arr.append(temp)
        init_arr = sorted(random_arr)
        print('---------------------------random_arr-----------------------:', init_arr)
        # print('x0:', x0)
        xy_min, func_min, index_dimen = func_model(index_ij=init_arr, x_min=x_ending.copy(), func_min=f0, max_iter_i=20,
                                                   nstart_i=10)
        print('xy_min:', xy_min)

        '''
        xy_min为当前最优值，已知最优值，再加上最新优化的几个维度，这种情况有可能会陷入局部最值，如何跳出局部最值，我们采用以一定概率P随机选择，非优化维度的最优坐标的做法

        '''
        pro = 0.30  # 我们设置跳出概率为0.3
        rand_pro = round(random.uniform(0, 1), 2)
        print('rand_pro:', rand_pro)
        print('index_dimen:', index_dimen)

        if rand_pro > pro:  # 采用当前最佳组合
            x_ending_temp = xy_min
            f_ending_temp = f_objective.f(x_ending_temp)
            if f_ending_temp < f_ending:
                x_ending = x_ending_temp
                f_ending = f_ending_temp

        #  当满足概率要求时，就随机选择一些点，作为最优值， 这样可以跳出局部解
        else:  # 随机数填充
            listxy = np.copy(xy_min)
            for index in range(f_objective.input_dim):
                if index not in index_dimen:  # 将非本次优化的维度的最优值点用取值区间内的随机数填充
                    listxy[index] = round(random.uniform(x_round[index][0], x_round[index][1]), 8)
            x_ending_temp = listxy

            f_ending_temp = f_objective.f(x_ending_temp)
            if f_ending_temp < f_ending:
                x_ending = x_ending_temp
                f_ending = f_ending_temp

        f_list.append(f_ending)

        print('历史记录每次迭代的最优值')
    print('f_list:', f_list)

    return f_list


# 单次迭代的最大次数
single_iter = 100

# 为了求均值，一共迭代了多少次
average_iter = 20

f_average = np.zeros((single_iter))

for i in range(average_iter):
    f_array = f_bo(num_iter=single_iter)
    f_average += np.array(f_array)

f_average = f_average / average_iter

plt.figure()

plt.plot(f_average, 'b')
plt.plot(f_average, 'ro')
plt.xlabel('the number of iters')
plt.ylabel('the max value of func')
plt.title('the func of SW(not jump 10 Dimension)')
new_xticks = np.linspace(0, single_iter,  11)
plt.xticks(new_xticks)
plt.savefig('../results_not_jump/SW_not_jump_10.jpg')
plt.show()

df_1 = pd.DataFrame(data=f_average)
df_1.to_csv('../results_not_jump/SW_not_jump_10.csv', sep='\t', header=None)

print('------------------------the ending results_jump------------------')
print(f_average)
