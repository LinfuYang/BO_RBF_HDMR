from HDMR import RBF_HDMR
from Objective_function import Func_3d
from collections import OrderedDict
import numpy as np
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import matern52
import warnings

f_objective = Func_3d.x1x2x3()
rbf_hdmr = RBF_HDMR.Rbf_Hdmr(f_objective)

# 其变量取值区间
x_round = f_objective.bounds

# 计算x0,f0。第一种选择方法，随机选取

x0, f0 = rbf_hdmr.x0_fx0()

point_round, f_values, type_fx, xishu_arr = rbf_hdmr.func_DEMO()



# 查看一阶模型的精度情况
is_True = rbf_hdmr.simulation_func()
print(is_True)

if is_True == False:
    # 由于我们规定，任何一种变量只能与其他某一种变量发生一次相关系，即 没一个变量的下标在x_ij_index数组中最多可以出现一次
    x_ij_index, x_ij_point, x_ij_value, x_ij_xishu = rbf_hdmr.f_Two_index()

    # 先判断哪些自变量是与其他自变量无关的
    # 整合相关自变量的数组
    x_inter = []
    for index in range(len(x_ij_index)):
        x_inter.append(list(x_ij_index[index]))

    x_inter = np.unique(x_inter)

    print('*********************************函数模型搭建完毕************************************')


def func_first_order(type_f=None, xishu_f=None, point_f=None):
    # 独立变量的线性、非线性判断
    if type_f == 'linear':
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
            x_min = x_round[1][0]

    # 独立变量的非线性情况
    else:
        # 非一维线性函数最好的办法采用BO来找函数最小值
        def f(x):
            return -(rbf_hdmr.func_1D_value(x, type=type_f, xishu=xishu_f, point_sample=point_f))

        sexp = matern52()
        gp = GaussianProcess(sexp)
        acq = Acquisition(mode='UCB')
        round_x = (x_round[0][0], x_round[0][1])
        param = {'x': ('cont', round_x)}
        gpgo = GPGO(gp, acq, f, param)
        gpgo.run(max_iter=5, nstart=1)
        res, f_min_i = gpgo.getResult()
        x_min = res['x']

    return x_min, f_min_i



def func_model(index_ij=None, init_x=None):
    '''
    :param index_ij: 选择的函数项
    :param init_x: 初始的函数最优解
    :return: 本地迭代函数最优解
    '''
    # 定义函数最小值
    func_min = 0
    # 函数最小值所在的坐标
    x_min = init_x
    first_order = []
    second_order = []
    for k in index_ij:
        if k < len(type_fx):
            # 一阶
            first_order.append(k)
        else:
            # 二阶
            second_order.append(k)

    # print(first_order)
    # print(second_order)
    '''
    
    #判断对应的坐标
    if len(first_order) == 0:
        # 只存在二位情况
        ij_index_0 = x_ij_index[second_order[0] - len(type_fx)]
        ij_xishu_0 = x_ij_xishu[second_order[0] - len(type_fx)]
        ij_point_0 = x_ij_point[second_order[0] - len(type_fx)]

        ij_index_1 = x_ij_index[second_order[1] - len(type_fx)]
        ij_xishu_1 = x_ij_xishu[second_order[1] - len(type_fx)]
        ij_point_1 = x_ij_point[second_order[1] - len(type_fx)]

        ij_index_2 = x_ij_index[second_order[2] - len(type_fx)]
        ij_xishu_2 = x_ij_xishu[second_order[2] - len(type_fx)]
        ij_point_2 = x_ij_point[second_order[2] - len(type_fx)]

        ij_index = np.unique([ij_index_0, ij_index_1, ij_index_2])
        print(ij_index)

    '''


func_model(index_ij=[5, 6, 14], init_x=x0)
























