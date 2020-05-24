import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import sys
import pprint as pp
import numpy.random as random

import sympy as sp

sys.path.append("../")
import custom_tools.fftplot as fftplot
import custom_tools.handyfuncs as hf

import control as con
import control.matlab as mctrl

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def s_plane_plot(sfunc, limits=[3, 3, 10], nsamp=500):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # ax.set_yscale('log')
    # ax.set_xscale('log')

    sigma = np.linspace(-limits[0], limits[0], nsamp)
    omega = sigma.copy()

    omega_1 = np.linspace(-limits[0], limits[0], nsamp)
    sigma_1 = np.zeros(sigma.size)

    s_zero_sig = sigma_1 + 1j * omega_1
    # plot 3D only takes in 1D arrays, 3 of them total
    ax.plot3D(sigma_1, omega_1, hf.db(sfunc(s_zero_sig)), 'r')

    sigma, omega = np.meshgrid(sigma, omega)
    s = sigma + 1j * omega
    # surface plot takes in 2D arrays, 3 of them total
    surf = ax.plot_surface(sigma, omega, hf.db(sfunc(s)), cmap=plt.cm.coolwarm)

    # cset = ax.contour(sigma, omega, hf.db(sfunc(s)), zdir='x')

    ax.set_zlim(-30, limits[2])

    plt.xlabel('$\sigma$')
    plt.ylabel('$j\omega$')
    fig.tight_layout()


def s_plot_val_func(num, den, s_para):
    num = np.squeeze(num)
    den = np.squeeze(den)
    n_array = []
    num_len = num.size
    for i in reversed(range(num_len)):
        n_array.append(s_para ** (i))

    num_sym = sum(np.multiply(num, n_array))

    d_array = []
    den_len = den.size
    for i in reversed(range(den_len)):
        d_array.append(s_para ** (i))

    den_sym = sum(np.multiply(den, d_array))

    return num_sym / den_sym


# Handles both s and z domain
# - z domain default Fs is 1
# - s domain fstop can be set through fstop_c

T = np.pi / 100  # sampling period
Fs = 1 / T
print('Sampling rate is {}Hz'.format(1 / T))
z = con.tf('z')
zm1 = 1 / z
# sys_und_tst = z / (z - 1)
# sys_und_tst = 1 / (z - 1)

# sys_und_tst = 1 + z ** -1 + z ** -2

s = con.tf('s')

# sys_und_tst = (s) / (1 + s)

# sys_und_tst = 1 / (s ** 3 + 2 * s ** 2 + 2 * s)

x = sp.symbols('x')
# x = 1


R1 = 9e6
C1 = 100e-12

R2 = 1e6
C2 = 110e-12

C3 = 50e-12

Z1 = R1 / (1 + s * R1 * C1)

Z2 = R2 / (1 + s * R2 * C2)

# Z2 = 1 / (s * C3) * (R2 + 1 / (s * C2)) / (1 / (s * C3) + (R2 + 1 / (s * C2)))

sys_und_tst = Z2 / (Z1 + Z2)

print(sys_und_tst)

sim_sys = con.minreal(sys_und_tst)

print(sim_sys)

sys_und_tst = sim_sys

gg = s_plot_val_func(sys_und_tst.num, sys_und_tst.den, x)

# https://www.sympy.org/scipy-2017-codegen-tutorial/notebooks/22-lambdify.html
# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
g = sp.lambdify([x], gg)

s_plane_plot(g, limits=[-10e3, 10e3, 10], nsamp=500)

fstop_c = 10e3

# impulse response
t_cd, im_resp = con.impulse_response(sys_und_tst)

plt.figure()
plt.plot(t_cd, im_resp, '-o')
plt.title("Impulse Response")
plt.xlabel("Time [s]")
plt.ylabel("Magnitude")
plt.grid()

# plot frequency response log scale
if sys_und_tst.isdtime():
    w = 2 * np.pi * np.logspace(-3, np.log10(0.5), 100)
else:
    w = 2 * np.pi * np.logspace(-3, np.log10(fstop_c), 100)

mag, phase, w = con.freqresp(sys_und_tst, w)
# freq response returns mag and phase as [[[mag]]], [[[phase]]]
# squeeze reduces this to a one dimensional array, optionally can use mag[0][0]
mag = np.squeeze(mag)
phase = np.squeeze(phase)
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogx(w / (2 * np.pi), hf.db(mag))
plt.grid()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.title('Frequency Response')

plt.subplot(2, 1, 2)
plt.semilogx(w / (2 * np.pi), np.unwrap(phase) * 180 / np.pi)
plt.grid()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [Deg]')

# plot frequency response linear scale
# beware the difference between linspace and logspace and that we're using only 100 points total
# phase plot in linear scale with 100pts will be off from logscale
if sys_und_tst.isdtime():
    w = 2 * np.pi * np.linspace(0, 0.5, 100)
else:
    w = 2 * np.pi * np.linspace(0, fstop_c, 100)

mag, phase, w = con.freqresp(sys_und_tst, w)
# freq response returns mag and phase as [[[mag]]], [[[phase]]]
# squeeze reduces this to a one dimensional array, optionally can use mag[0][0]
mag = np.squeeze(mag)
phase = np.squeeze(phase)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w / (2 * np.pi), hf.db(mag))
plt.grid()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.title('Frequency Response Linear scale')

plt.subplot(2, 1, 2)
plt.plot(w / (2 * np.pi), np.unwrap(phase) * 180 / np.pi)
plt.grid()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [Deg]')

# pole zero plot
poles, zeros = con.pzmap(sys_und_tst)
plt.axis('equal')

print(f'ploes are {poles}')
print(f'zeros are {zeros}')

# plt.figure()
# plt.grid()
# plt.axis('equal')
# if len(poles) > 0:
#     plt.plot(poles.real, poles.imag, 'bx')
# if len(zeros) > 0:
#     plt.plot(zeros.real, zeros.imag, 'bo')

if sys_und_tst.isdtime():
    cir_phase = np.linspace(0, 2 * np.pi, 500)
    plt.plot(np.real(np.exp(1j * cir_phase)), np.imag(np.exp(1j * cir_phase)), 'r--')

plt.show()
