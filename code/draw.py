import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op


def func(x, a, b, c):
    return a / x ** c + b


x = [01.00E-07, 1.00E-07, 1.00E-07, 1.00E-07, 1.00E-07, 1.00E-07, 1.00E-07, 0.0342, 0.2789, 0.5532, 0.593, 0.6558,
     0.7561, 0.812,
     0.8667,
     1,
     1,
     1,
     1,
     1,
     1]
y = [0.95,
     0.95,
     0.95,
     0.9356,
     0.866,
     0.5176,
     0.3614,
     0.0654,
     0.0105,
     0.0033,
     0.0016,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     ]
x = np.array(x)
y = np.array(y)
plt.scatter(x, y, color='deepskyblue')
a, b, c = op.curve_fit(func, x, y)[0]
print(a, b, c)
x = np.arange(0.00000001, 1, 0.00000001)
y = func(x, a, b, c)
vit_no, = plt.plot(x, y, color='deepskyblue')

x = [1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     0.0743,
     0.2333,
     0.2497,
     0.2558,
     0.2681,
     0.2909,
     0.3242,
     0.3506,
     0.4969,
     0.9041,
     1,
     1,
     1,
     1
     ]
y = [0.95,
     0.95,
     0.95,
     0.9428,
     0.9059,
     0.5745,
     0.3748,
     0.0918,
     0.0405,
     0.0072,
     0.0068,
     0.0058,
     0.0049,
     0.0029,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07
     ]
x = np.array(x)
y = np.array(y)
plt.scatter(x, y, color='red')
a, b, c = op.curve_fit(func, x, y)[0]
print(a, b, c)
x = np.arange(0.00000001, 1, 0.00000001)
y = func(x, a, b, c)
vit_yes, = plt.plot(x, y, color='red')

x = [1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     1.00E-07,
     0.0067,
     0.2962,
     0.3006,
     0.3082,
     0.3523,
     0.926,
     1,
     ]
y = [0.95,
     0.95,
     0.95,
     0.95,
     0.95,
     0.95,
     0.95,
     0.948,
     0.9363,
     0.9203,
     0.9173,
     0.9173,
     0.9137,
     0.9085,
     0.9052,
     0.8837,
     0.8791,
     0.8748,
     0.8519,
     0.8042,
     0.4454,
     0.4454,
     0.3101,
     0.0556,
     0.0059,
     0.0059,
     0.0059,
     0.0039,
     1e-7,
     1e-7
     ]

x = np.array(x)
y = np.array(y)
plt.scatter(x, y, color='green')
a, b, c = op.curve_fit(func, x, y)[0]
print(a, b, c)
x = np.arange(0.00000001, 1, 0.00000001)
y = func(x, a, b, c)
pca, = plt.plot(x, y, color='green')
plt.xlabel('FAR')
plt.ylabel('FRR')
plt.legend(handles=[vit_no, vit_yes, pca], labels=["ViT", "ViT Augment", "PCA"], loc="upper right", fontsize=6)
plt.savefig('lines.png')
plt.show()
