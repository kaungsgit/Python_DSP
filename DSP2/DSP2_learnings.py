import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.fftpack as fft

import sys

sys.path.append("../")

import custom_tools.fftplot as fftplot
# import src.gps.gps as gps
from numpy.random import randn

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def s_plane_plot(sfunc, limits=[3, 3, 10], nsamp=500):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    sigma = np.linspace(-limits[0], limits[0], nsamp)
    omega = sigma.copy()

    sigma, omega = np.meshgrid(sigma, omega)

    s = sigma + 1j * omega

    surf = ax.plot_surface(sigma, omega, np.abs(sfunc(s)), cmap=cm.flag)
    ax.set_zlim(0, limits[2])
    plt.xlabel('$\sigma$')
    plt.ylabel('$j\omega$')
    fig.tight_layout()


s_plane_plot(lambda s: 1 / s, nsamp=50)


def X(s):
    return 1 / ((s + .2 + .5j) * (s + .2 - .5j))


s_plane_plot(X, limits=[1, 1, 4], nsamp=40)

plt.show()
