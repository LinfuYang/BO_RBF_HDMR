import numpy as np

import matplotlib.pyplot as plt


def f(x):
    return  (x * np.sin(x) + 0.1 * x)

# 输出任意函数值
print('f(0)', f(-0.05))


X = np.linspace(-0.2, 0.2, 100)
y = []

for i in X:
    y.append(f(i))

plt.figure()
plt.plot(X, y)
plt.show()

# print('x_point：', X)
# print('f_value：', y)
#
#
