import numpy as np
import numpy.linalg as la
import scipy.signal as sig
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import sympy as sp

x_max = 80
num_poly = 5

# y = np.mat([np.random.randint(50)+10*i for i in range(x_max)]).T
# x = np.mat([i for i in range(x_max)]).T
# y = 2 * np.sin(0.05 * np.pi * x)

x = np.mat([-10, -5, 0, 5, 10, 15, 20, 25], dtype=np.double).T
y = np.mat([0, 5.5, 10, 14, 18.5, 21, 21, 21], dtype=np.double).T

plt.plot(x, y)
ones_list = np.mat(np.ones(len(x))).T
# F = np.column_stack((ones_list, x))
zz = [np.power(x, i) for i in range(num_poly)]
F = np.column_stack(zz)
# a = np.dot(la.inv(F), y)
a = np.linalg.inv(F.T @ F) @ F.T @ y

x1, y1 = sp.symbols('x1 y1')
# y1 = a.A[1][0]*x1+a.A[0][0]

k = sp.symbols('k')

y1 = 0
for index, val in enumerate(a.flat):
    y1 = y1 + val * x1 ** index

x_vals = np.arange(x[0], x[-1], (x[-1]-x[0])/15)
y_vals = [y1.subs(x1, i) for i in x_vals]

plt.figure()
plt.plot(x_vals, y_vals)
plt.plot(x, y, 'bo')

plt.show()

pass
