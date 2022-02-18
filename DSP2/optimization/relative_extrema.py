import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# x = sp.symbols('x')
# y = sp.symbols('y')
#
# some_eq = x ** 2 - 6 * x - 8 * y + y ** 2


def f(x, y):
    # return np.sin(np.sqrt(x ** 2 + y ** 2))
    return x ** 2 - 6 * x - 8 * y + y ** 2


print(f(3, 4))

x = np.linspace(-0, 5, 60)
y = np.linspace(-0, 5, 60)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
