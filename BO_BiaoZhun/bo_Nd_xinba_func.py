from Objective_function import Func_Nd
import numpy as np
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import matern52
import warnings
warnings.filterwarnings('ignore')

# 函数
f_objective = Func_Nd.Schwefel_Func(input_dim=10)
x_round = f_objective.bounds

input_dim = f_objective.input_dim


# 一阶函数的最小值
# 数量

X = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
     'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
X_name = X[:input_dim]



def f(X_name):
    return -f_objective.f(X_name)

def f_bo(single_iter_bo=100):

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
    gpgo.run(max_iter=single_iter_bo, nstart=100)
    res, f_min_xy = gpgo.getResult()
    f_list = []
    f_list.extend(gpgo.return_max_f())
    print('f_list:', f_list)
    return f_list


single_iter = 100
average_iter = 20
f_average = np.zeros(single_iter+1)


for i in range(average_iter):
    f_array = f_bo(single_iter_bo=single_iter)
    f_average += np.array(f_array)

f_average = -1 * f_average / average_iter

plt.figure()

plt.plot(f_average, 'b')
plt.plot(f_average, 'ro')
plt.xlabel('the number of iters')
plt.ylabel('the max value of func')
plt.title('the func of SW(Bo 10 Dimensions)')
new_xticks = np.linspace(0, single_iter,  11)
plt.xticks(new_xticks)
plt.savefig('../results_normal/SW_bo_10.jpg')
plt.show()


df_1 = pd.DataFrame(data=f_average)
df_1.to_csv('../results_normal/SW_bo_10.csv', sep='\t')


print('f_average:', f_average)




