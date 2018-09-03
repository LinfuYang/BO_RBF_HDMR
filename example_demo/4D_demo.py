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
print(x0)
print(f0)

point_round, f_values, type_fx, xishu_arr = rbf_hdmr.func_DEMO()

print(point_round)
print(f_values)
print(type_fx)
print(xishu_arr)
# 查看一阶模型的精度情况
is_True = rbf_hdmr.simulation_func()
# print(is_True)

if is_True == False:
    # 由于我们规定，任何一种变量只能与其他某一种变量发生一次相关系，即 没一个变量的下标在x_ij_index数组中最多可以出现一次
    x_ij_index, x_ij_point, x_ij_value, x_ij_xishu = rbf_hdmr.f_Two_index()
    print(x_ij_index)
    print(x_ij_point)
    print(x_ij_value)
    print(x_ij_xishu)


