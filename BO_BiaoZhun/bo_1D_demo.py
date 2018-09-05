from Objective_function import Func_1d

from collections import OrderedDict
import numpy as np
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import matern52
import warnings
warnings.filterwarnings('ignore')

# 实例化对象
f_objective = Func_1d.test_1d_func()
# 自变量的取值区间
x_round = f_objective.bounds


X = ['A']
def f (X):
    return - f_objective.f(X)

sexp = matern52()
gp = GaussianProcess(sexp)
acq = Acquisition(mode='ExpectedImprovement')

param = OrderedDict()
for i in range(len(X)):
    param[X[i]] = ('cont', x_round[i])


gpgo = GPGO(gp, acq, f, param)
gpgo.run(max_iter=100, nstart=20)
res, f_min_xy = gpgo.getResult()

print(res)
print(f_min_xy)



