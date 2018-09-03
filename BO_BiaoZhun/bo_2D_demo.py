from Objective_function import Func_2d

from collections import OrderedDict
import numpy as np
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import matern52
import warnings
warnings.filterwarnings('ignore')

# 函数
f_objective = Func_2d.branin()
x_round = f_objective.bounds




# 一阶函数的最小值
# 数量

X = ['A', 'B']
def f(X):

    return -f_objective.f(X)
sexp = matern52()
gp = GaussianProcess(sexp)
'''
    'ExpectedImprovement': self.ExpectedImprovement,
    'IntegratedExpectedImprovement': self.IntegratedExpectedImprovement,
    'ProbabilityImprovement': self.ProbabilityImprovement,
    'IntegratedProbabilityImprovement': self.IntegratedProbabilityImprovement,
    'UCB': self.UCB,
    'IntegratedUCB': self.IntegratedUCB,
    'Entropy': self.Entropy,
    'tExpectedImprovement': self.tExpectedImprovement,
    'tIntegratedExpectedImprovement': self.tIntegratedExpectedImprovement
'''
acq = Acquisition(mode='ExpectedImprovement')
param = OrderedDict()
for i in range(len(X)):
    param[X[i]] = ('cont', x_round[i])

gpgo = GPGO(gp, acq, f, param)
gpgo.run(max_iter=1, nstart=100)
res, f_min_xy = gpgo.getResult()

print(res)
print(f_min_xy)


f_ture = f_objective.fmin
print('最小值点：', res)


print('原函数精确值最小值：', f_ture)
print('一阶函数模型最小值：', f_min_xy)


if f_ture != 0:
    print('原函数精确值最小值：', f_ture)
    print('函数近似模型最小值：', f_min_xy)
    corr_err = abs((f_ture - f_min_xy) / f_ture) * 100
    print('corr_err:', corr_err)




