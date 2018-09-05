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
f_objective = Func_Nd.Gaussian_mixture_function(input_dim=5)
x_round = f_objective.bounds

X = [2] * 5
print(-f_objective.f(X))


