"""
This demonstrates how a polynomial regression can be implemented without using Linear Algebra.
Spoiler Alert! It's very tedious!
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import randn

# A/D Conversion, Sampling
nsamps = 20
# generate time vector
# fs = 500e6
# ftone = 10e6
# t = np.arange(nsamps) * 1 / fs
T = 1
# n = np.arange(0, nsamps)
sum_end = 80
num_terms_x_est = 10  # number of coeffs to be used for x_est_idx, x_est is always fixed at 5
c = sp.IndexedBase('c')
j, k, n = sp.symbols('j k n')
c1, c2, c3, c4, c5 = sp.symbols('c1 c2 c3 c4 c5')
w0 = sp.symbols('w0')

# n = sp.symbols('n')

x_truth = 2 * sp.sin(0.05 * np.pi * n)
# x_truth = n
# x_est = c1 + c2 * n * T + c3 * (n * T) ** 2
# x_est = c1 * sp.cos(w0 * n)

x_est = c1 + c2 * n * T + c3 * (n * T) ** 2 + c4 * (n * T) ** 3 + c5 * (n * T) ** 4  # num of coeffs fixed at 5
x_est_idx = (sp.Sum(c[k] * n ** (k - 1), (k, 1, num_terms_x_est))).doit()

J = sp.concrete.Sum((x_truth - x_est) ** 2, (n, 1, sum_end))

J_1 = (sp.Sum((x_truth - x_est_idx) ** 2, (n, 1, sum_end)))

# @todo: indexed equations with derivatives
# https://stackoverflow.com/questions/37647370/expand-index-notation-equation-using-sympy
# https://stackoverflow.com/questions/26402387/sympy-summation-with-indexed-variable
# https://stackoverflow.com/questions/48479317/what-exactly-are-indexed-objects-in-sympy

# from sympy import Sum, symbols, Indexed, lambdify
# import numpy as np
#
# x, i = symbols("x i")
# s = Sum(Indexed('x',i),(i,0,3))
# f = lambdify(x, s)
# b = np.array([1, 2, 3, 4])
# f(b)

# from sympy import *
# x = IndexedBase('x')
# j, k, n = symbols('j k n', cls=Idx)
# f = 1/sqrt(Sum(x[k]**2, (k, 1, n)))
# print(f.diff(x[j]))

eq1 = sp.diff(J, c1).doit()
eq2 = sp.diff(J, c2).doit()
eq3 = sp.diff(J, c3).doit()
eq4 = sp.diff(J, c4).doit()
eq5 = sp.diff(J, c5).doit()

res = sp.solve([eq1, eq2, eq3, eq4, eq5], (c1, c2, c3, c4, c5))
x_est_1 = x_est.subs([(c1, res[c1]),
                      (c2, res[c2]),
                      (c3, res[c3]),
                      (c4, res[c4]),
                      (c5, res[c5])])

c_list = [c[i] for i in range(1, num_terms_x_est + 1)]
eq_list = [sp.diff(J_1, c[i]).doit() for i in range(1, num_terms_x_est + 1)]
res_1 = sp.solve(eq_list, c_list)
x_est_idx_1 = x_est_idx.subs(res_1)

p1 = sp.plot(x_est_1, x_truth, (n, 0, sum_end))

p2 = sp.plot(x_est_idx_1, x_truth, (n, 0, sum_end))

pass
