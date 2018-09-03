from HDMR import RBF_HDMR
from Objective_function import Func_3d
from collections import OrderedDict
import numpy as np
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import matern52
import warnings


f_objective = Func_3d.lizi_3d()
rbf_hdmr = RBF_HDMR.Rbf_Hdmr(f_objective)
# 其变量取值区间
x_round = f_objective.bounds

# 计算x0,f0。第一种选择方法，随机选取

x0, f0 = rbf_hdmr.x0_fx0()
print(x0)
print(f0)

point_round, f_values, type_fx, xishu_arr = rbf_hdmr.func_DEMO()

print(point_round)
print(f_values)
print(type_fx)
print(xishu_arr)

# 查看一阶模型的精度情况
is_True = rbf_hdmr.simulation_func()

if is_True == False:
    # 由于我们规定，任何一种变量只能与其他某一种变量发生一次相关系，即 没一个变量的下标在x_ij_index数组中最多可以出现一次
    x_ij_index, x_ij_point, x_ij_value, x_ij_xishu = rbf_hdmr.f_Two_index()

    # 先判断哪些自变量是与其他自变量无关的
    # 整合相关自变量的数组
    x_inter = list([])
    for index in range(len(x_ij_index)):
        x_inter.append(list(x_ij_index[index]))
        x_inter = np.unique(x_inter)
    print(x_ij_index)
    # print(x_ij_point)
    # print(x_ij_value)
    # print(x_ij_xishu)
print('*********************************函数模型搭建完毕************************************')




f_min_value = f0
round_xyz = np.zeros((len(x0)))
num_fun_1D = len(type_fx)

for i in range(num_fun_1D):

    if is_True == False:

        # 说明一阶函数模型无法满足精度要求，即存在两两相关的自变量
        if i not in x_inter:

            type_fx_i = type_fx[i]
            xishu_arr_i = xishu_arr[i]
            point_round_i = point_round[i]

            if type_fx_i == 'linear':
                # 左端点函数值
                f_left = rbf_hdmr.func_1D_value(x_round[i][0], type=type_fx_i, xishu=xishu_arr_i,
                                                point_sample=point_round_i)
                # 右端点函数值
                f_right = rbf_hdmr.func_1D_value(x_round[i][1], type=type_fx_i, xishu=xishu_arr_i,
                                                 point_sample=point_round_i)
                if f_left > f_right:
                    f_min_i = f_right
                    round_xyz.append(x_round[i][1])
                else:
                    f_min_i = f_left
                    round_xyz.append(x_round[i][0])
                print('f_min_i:', f_min_i[0])
                f_min_value += f_min_i[0]

            else:
                # 非一维线性函数最好的办法采用BO来找函数最小值
                def f(x):
                    return -(rbf_hdmr.func_1D_value(x, type=type_fx_i, xishu=xishu_arr_i, point_sample=point_round_i))

                sexp = matern52()
                gp = GaussianProcess(sexp)
                acq = Acquisition(mode='UCB')
                round_x = (x_round[i][0], x_round[i][1])
                param = {'x': ('cont', round_x)}
                gpgo = GPGO(gp, acq, f, param)
                gpgo.run(max_iter=5, nstart=1)
                res, max_y = gpgo.getResult()
                round_xyz[i] = res['x']
                print('i:', i)
                print(max_y)
                f_min_value += max_y


        else:
            # 说明该变量存在有且仅有一个相关变量
            for index in range(len(x_ij_index)):
                if i == x_ij_index[index, 0]:

                    # 一阶模型情况 i
                    print('存在相关联变量**********************************************', i)
                    type_fx_i = type_fx[i]
                    xishu_arr_i = xishu_arr[i]
                    point_round_i = point_round[i]

                    # 一阶情况 j
                    j = x_ij_index[index, 1]
                    type_fx_j = type_fx[j]
                    xishu_arr_j = xishu_arr[j]
                    point_round_j = point_round[j]


                    # 二阶情况
                    ij_index = x_ij_index[index]
                    ij_xishu = x_ij_xishu[index]
                    ij_point = x_ij_point[index]

                    def f(x, y):
                        f_x_1 = - rbf_hdmr.func_1D_value(x, type=type_fx_i, xishu=xishu_arr_i,
                                                            point_sample=point_round_i)
                        f_x_2 = - rbf_hdmr.func_1D_value(x, type=type_fx_j, xishu=xishu_arr_j,
                                                         point_sample=point_round_j)

                        f_2 = -(rbf_hdmr.func_2D_value(x, y, index_ij=ij_index, xishu=ij_xishu, points=ij_point))
                        return f_x_1 + f_x_2 + f_2
                    param = OrderedDict()
                    param['x'] = ('cont', x_round[ij_index[0]])
                    param['y'] = ('cont', x_round[ij_index[1]])
                    # print(param)
                    sexp = matern52()
                    gp = GaussianProcess(sexp)
                    acq = Acquisition(mode='ExpectedImprovement')
                    gpgo = GPGO(gp, acq, f, param)
                    gpgo.run(max_iter=100, nstart=10)
                    res, max_xy = gpgo.getResult()
                    round_xyz[ij_index[0]] = res['x']
                    round_xyz[ij_index[1]] = res['y']
                    print('i:', ij_index[0])
                    print('j:', ij_index[1])
                    print(max_xy)
                    f_min_value += max_xy

print('round_xyz:', round_xyz)
print('f_min_value:', f_min_value)


















