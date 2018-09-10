import numpy as np

import matplotlib.pyplot as plt


f_average = [7.56263865e+00,   7.69721359e-01,   7.68973686e-01,   1.83859567e-04,
             1.83859567e-04,   1.21110017e-04,   1.21110017e-04,   1.13526807e-04,
             1.02234680e-04,   1.02234680e-04,   1.02234680e-04,   1.00349328e-04,
             1.00349328e-04,   1.00349328e-04,   1.00349328e-04,   1.00349328e-04,
             1.00349328e-04,   1.00349328e-04,   9.00364090e-05,   9.00364090e-05]

plt.figure()

plt.plot(f_average, 'b')
plt.plot(f_average, 'ro')
new_xticks = np.linspace(0, 20, 7)
plt.xticks(new_xticks)
plt.xlabel('the number of iters')
plt.ylabel('the max value of func')
plt.title('the func of SW(not jump)')
plt.savefig('../results_not_jump/SW_not_jump_10.jpg')
plt.show()