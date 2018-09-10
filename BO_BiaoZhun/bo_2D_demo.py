from Objective_function import Func_Nd

from collections import OrderedDict
import numpy as np
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import matern52
import warnings
warnings.filterwarnings('ignore')

# 函数
dim = 10
f_objective = Func_Nd.Schwefel_Func(input_dim=dim)
x_round = f_objective.bounds

X2 = [0.5] * dim
print(f_objective.f(X2))


