from Objective_function import Func_3d

from collections import OrderedDict
import numpy as np
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import matern52
import warnings
warnings.filterwarnings('ignore')

# 函数
f_objective = Func_3d.lizi_3d()

x_round = f_objective.bounds

input_dim = f_objective.input_dim


# 一阶函数的最小值
# 数量

X = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
     'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
X_name = X[:input_dim]
def f(X_name):
    return -f_objective.f(X_name)
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
for temp in X_name:
    param[temp] = ('cont', x_round[0])
gpgo = GPGO(gp, acq, f, param)
gpgo.run(max_iter=200, nstart=100)
res, f_min_xy = gpgo.getResult()




f_ture = f_objective.fmin



print('原函数精确值最小值：', f_ture)
print('函数最小值输入变量：', res)
print('函数最小值近似结果：', f_min_xy)


if f_ture != 0:
    corr_err = abs((f_ture - f_min_xy) / f_ture) * 100
    print('corr_err:', corr_err)




