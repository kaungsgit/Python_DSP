"""
@author: Kaung Myat San Oo
Specify sys_und_tst in either s or z form.
Impulse response, frequency response, and pole-zero plots will be shown.
"""

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


# import plotly.express as px


def s_plane_plot(sys, sfunc, limits=[3, 3, 10], nsamp=500):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # ax.set_yscale('log')
    # ax.set_xscale('log')

    sigma = np.linspace(-limits[0], limits[0], nsamp)
    omega = sigma.copy()

    if sys.isdtime():

        # drawing z plane freq response
        cir_phase = np.linspace(0, 2 * np.pi, nsamp)

        sigma_x = np.real(np.exp(1j * cir_phase))
        omega_y = np.imag(np.exp(1j * cir_phase))

        z_zero_omega = sigma_x + 1j * omega_y

        freq_resp_dB = hf.db(sfunc(z_zero_omega))
        # plot 3D only takes in 1D arrays, 3 of them total
        ax.plot3D(sigma_x, omega_y, freq_resp_dB, 'r')
        plt.xlabel('Real z')
        plt.ylabel('Imag z')
    else:
        omega_1 = np.linspace(-limits[0], limits[0], nsamp)
        sigma_1 = np.zeros(sigma.size)

        s_zero_sig = sigma_1 + 1j * omega_1

        freq_resp_dB = hf.db(sfunc(s_zero_sig))
        # plot 3D only takes in 1D arrays, 3 of them total
        ax.plot3D(sigma_1, omega_1, freq_resp_dB, 'r')
        plt.xlabel('$\sigma$')
        plt.ylabel('$j\omega$')

    sigma, omega = np.meshgrid(sigma, omega)
    s = sigma + 1j * omega
    # surface plot takes in 2D arrays, 3 of them total
    surf = ax.plot_surface(sigma, omega, hf.db(sfunc(s)), cmap=plt.cm.coolwarm)

    # cset = ax.contour(sigma, omega, hf.db(sfunc(s)), zdir='x')

    ax.set_zlim(-30, limits[2])
    # ax.set_xlim(limits[0] / (2 * np.pi), -limits[0] / (2 * np.pi))
    # ax.set_ylim(limits[0] / (2 * np.pi), -limits[0] / (2 * np.pi))

    fig.tight_layout()

    return freq_resp_dB


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
# - s domain fstop can be set through fstop_c_log_scale

# T = np.pi / 100  # sampling period
# Fs = 1 / T

Fs = 1e6
T = 1 / Fs

print('Sampling rate is {}Hz'.format(1 / T))
z = con.tf('z')
zm1 = 1 / z

s = con.tf('s')

x = sp.symbols('x')
# x = 1

# continuous

R1 = 9e6
C1 = 12.2e-12
R2 = 1e6
C2 = 110e-12
C3 = 50e-12

Z1 = R1 / (1 + s * R1 * C1)
Z2 = R2 / (1 + s * R2 * C2)
# Z2 = 1 / (s * C3) * (R2 + 1 / (s * C2)) / (1 / (s * C3) + (R2 + 1 / (s * C2)))

sys_und_tst = (s) / (1000 + s)
# sys_und_tst = 1 / (s ** 3 + 2 * s ** 2 + 2 * s)
# sys_und_tst = Z2 / (Z1 + Z2)

# discrete
# sys_und_tst = 1 + z ** -1 # low pass fir filter
# sys_und_tst = 1 / (1 + 1e-3 / (z - 1))  # dc notch filter

# sys_und_tst = z / (z - 1)
# sys_und_tst = 1 / (z - 1)
# sys_und_tst = 1 + z ** -1 + z ** -2

print(sys_und_tst)

sim_sys = con.minreal(sys_und_tst)

print(sim_sys)

sys_und_tst = sim_sys

gg = s_plot_val_func(sys_und_tst.num, sys_und_tst.den, x)

# https://www.sympy.org/scipy-2017-codegen-tutorial/notebooks/22-lambdify.html
# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
g = sp.lambdify([x], gg)

fstop_c_log_scale = 1e6
# fstop_c_log_scale = 1e6

nsamp_pos_neg = 1000
nsamp_single = int(nsamp_pos_neg / 2)

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
    cir_phase = np.linspace(0, 2 * np.pi, nsamp_single)
    plt.plot(np.real(np.exp(1j * cir_phase)), np.imag(np.exp(1j * cir_phase)), 'r--')

fstop_c_linear_scale = 0
if sys_und_tst.isdtime():
    freq_resp_dB = s_plane_plot(sys_und_tst, g, limits=[-2, 2, 10], nsamp=nsamp_pos_neg)
else:
    # freq_resp_dB = s_plane_plot(sys_und_tst, g, limits=[-fstop_c_log_scale * 2 * np.pi, fstop_c_log_scale * 2 * np.pi, 10],
    #                             nsamp=nsamp_pos_neg)
    poles_zeros = np.concatenate((poles, zeros), axis=0)
    # fstop_c_linear_scale = min(poles_zeros) + 0.3 * min(poles_zeros)
    fstop_c_linear_scale = max(abs(poles_zeros)) + 0.5 * max(abs(poles_zeros))

    freq_resp_dB = s_plane_plot(sys_und_tst, g,
                                limits=[fstop_c_linear_scale * 2 * np.pi,
                                        fstop_c_linear_scale * 2 * np.pi, 10],
                                nsamp=nsamp_pos_neg)

    # zoomed in plot
    _ = s_plane_plot(sys_und_tst, g,
                     limits=[fstop_c_linear_scale,
                             fstop_c_linear_scale, 10],
                     nsamp=nsamp_pos_neg)

# impulse response
t_cd, im_resp = con.impulse_response(sys_und_tst)

plt.figure()
plt.plot(t_cd, im_resp, '-o')
plt.title("Impulse Response")
plt.xlabel("Time [s]")
plt.ylabel("Magnitude")
plt.grid()


# fig = px.line(x=t_cd, y=im_resp, labels={'x': 'Time [s]', 'y': 'Magnitude'})
# fig.show()

# bode plot details
# - w or f
# - log or linear x axis


class BodePlot:

    def __init__(self, is_angular_freq=False, is_xlog_scale=True, is_ylog_scale=False, title='Frequency Response'):
        self.is_angular_freq = is_angular_freq
        self.is_xlog_scale = is_xlog_scale
        self.is_ylog_scale = is_ylog_scale
        self.x_axis_label = 'Frequency [Hz]'
        self.mag_label = 'Magnitude [dB]'
        self.phase_label = 'Phase [Deg]'
        self.title = title
        pass

    def convert_to_f(self, w):
        if not self.is_angular_freq:
            converted_freq = w / (2 * np.pi)
            self.x_axis_label = 'Frequency [Hz]'
        else:
            converted_freq = w
            self.x_axis_label = 'Frequency [rad/s]'
        return converted_freq

    def convert_to_linear(self):
        if self.is_xlog_scale:
            return 'semilogx'
        else:
            return 'plot'

    def plot_(self, x, y):
        if self.is_xlog_scale and self.is_ylog_scale:
            plt.loglog(x, y)
        elif self.is_xlog_scale:
            plt.semilogx(x, y)
        elif self.is_ylog_scale:
            plt.semilogy(x, y)
        else:
            plt.plot(x, y)

    @staticmethod
    def calc_freq_response(sys_und_tst, w):
        mag, phase, w = con.freqresp(sys_und_tst, w)

        # freq response returns mag and phase as [[[mag]]], [[[phase]]]
        # squeeze reduces this to a one dimensional array, optionally can use mag[0][0]
        mag = np.squeeze(mag)
        phase = np.squeeze(phase)

        return mag, phase, w

    def plot(self, sys_und_tst, w, title=''):
        converted_freq = self.convert_to_f(w)

        mag, phase, w = self.calc_freq_response(sys_und_tst, w)

        plt.figure()
        plt.subplot(2, 1, 1)

        self.plot_(converted_freq, hf.db(mag))
        # plt.semilogx(w / (2 * np.pi), hf.db(mag))
        plt.grid()
        plt.xlabel(self.x_axis_label)
        plt.ylabel(self.mag_label)
        plt.title((lambda x: x if x != '' else self.title)(title))

        plt.subplot(2, 1, 2)
        self.plot_(converted_freq, np.unwrap(phase) * 180 / np.pi)
        # plt.semilogx(w / (2 * np.pi), np.unwrap(phase) * 180 / np.pi)
        plt.grid()
        plt.xlabel(self.x_axis_label)
        plt.ylabel(self.phase_label)


# plot frequency response log scale
start_exponent = -9  # may need to adjust this to see correct reponse of very narrow dc notch filters
if sys_und_tst.isdtime():
    w = 2 * np.pi * np.logspace(start_exponent, np.log10(0.5),
                                nsamp_single)
else:
    w = 2 * np.pi * np.logspace(start_exponent, np.log10(fstop_c_log_scale), nsamp_single)

bp = BodePlot(is_angular_freq=False, is_xlog_scale=True, is_ylog_scale=False)
bp.plot(sys_und_tst, w, title='Freq Response, freq on log scale')
# plot frequency response log scale, pi axis
start_exponent = -9  # may need to adjust this to see correct reponse of very narrow dc notch filters
if sys_und_tst.isdtime():
    w = 2 * np.pi * np.logspace(start_exponent, np.log10(0.5),
                                nsamp_single)
else:
    w = 2 * np.pi * np.logspace(start_exponent, np.log10(fstop_c_log_scale), nsamp_single)

bp = BodePlot(is_angular_freq=True, is_xlog_scale=True, is_ylog_scale=False)
bp.plot(sys_und_tst, w, title='Freq Response, omega on log scale')

# plot frequency response linear scale
# beware the difference between linspace and logspace and that we're using only 100 points total
# phase plot in linear scale with 100pts will be off from logscale
if sys_und_tst.isdtime():
    w = 2 * np.pi * np.linspace(0, 0.5, nsamp_single)
else:
    w = 2 * np.pi * np.linspace(0, abs(fstop_c_linear_scale), nsamp_single)

bp = BodePlot(is_angular_freq=False, is_xlog_scale=False)
bp.plot(sys_und_tst, w, title='Freq Response, freq on linear scale')

mag, phase, w = con.freqresp(sys_und_tst, w)
# freq response returns mag and phase as [[[mag]]], [[[phase]]]
# squeeze reduces this to a one dimensional array, optionally can use mag[0][0]
mag = np.squeeze(mag)
phase = np.squeeze(phase)

plt.figure()
if sys_und_tst.isdtime():
    plt.plot(w / (2 * np.pi), np.flip(freq_resp_dB[int(freq_resp_dB.size / 2):]), label='Freq response from z-plane')
else:
    plt.plot(w / (2 * np.pi), freq_resp_dB[int(freq_resp_dB.size / 2):], label='Freq response from s-plane')
plt.plot(w / (2 * np.pi), hf.db(mag), 'r--', label='Freq response from con.freqresp')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Mag [dB]')
plt.title('Comparing freq response from s/z plane to con.freqresp')
plt.legend()

plt.show()
