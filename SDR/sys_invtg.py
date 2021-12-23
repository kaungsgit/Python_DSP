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
# - s domain fstop can be set through fstop_c

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

fstop_c = 1e6
# fstop_c = 1e6

nsamp_pos_neg = 1000
nsamp_single = int(nsamp_pos_neg / 2)

if sys_und_tst.isdtime():
    freq_resp_dB = s_plane_plot(sys_und_tst, g, limits=[-2, 2, 10], nsamp=nsamp_pos_neg)
else:
    freq_resp_dB = s_plane_plot(sys_und_tst, g, limits=[-fstop_c * 2 * np.pi, fstop_c * 2 * np.pi, 10],
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
# plot frequency response log scale
start_exponent = -9  # may need to adjust this to see correct reponse of very narrow dc notch filters
if sys_und_tst.isdtime():
    w = 2 * np.pi * np.logspace(start_exponent, np.log10(0.5),
                                nsamp_single)
else:
    w = 2 * np.pi * np.logspace(start_exponent, np.log10(fstop_c), nsamp_single)

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

# plot frequency response log scale, pi axis
start_exponent = -9  # may need to adjust this to see correct reponse of very narrow dc notch filters
if sys_und_tst.isdtime():
    w = 2 * np.pi * np.logspace(start_exponent, np.log10(0.5),
                                nsamp_single)
else:
    w = 2 * np.pi * np.logspace(start_exponent, np.log10(fstop_c), nsamp_single)

mag, phase, w = con.freqresp(sys_und_tst, w)
# freq response returns mag and phase as [[[mag]]], [[[phase]]]
# squeeze reduces this to a one dimensional array, optionally can use mag[0][0]
mag = np.squeeze(mag)
phase = np.squeeze(phase)
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogx(w, hf.db(mag))
plt.grid()
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')
plt.title('Frequency Response')

plt.subplot(2, 1, 2)
plt.semilogx(w, np.unwrap(phase) * 180 / np.pi)
plt.grid()
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Phase [Deg]')

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# # with plotly, it looks good but takes too long...
# fig = make_subplots(rows=2, cols=1,
#                     shared_xaxes=True,
#                     vertical_spacing=0.02)
#
# fig.append_trace(go.Scatter(
#     x=w / (2 * np.pi), y=hf.db(mag),
# ), row=1, col=1)
#
# fig.append_trace(go.Scatter(
#     x=w / (2 * np.pi), y=np.unwrap(phase) * 180 / np.pi,
# ), row=2, col=1)
#
# fig.update_xaxes(title_text="Frequency [Hz]", type="log")
# fig.update_yaxes(title_text="Mag [dB]", row=1, col=1)
# fig.update_yaxes(title_text="Phase [Deg]", row=2, col=1)
#
# # fig.update_layout(height=600, width=600, title_text="Stacked Subplots")
# fig.show()

# plot frequency response linear scale
# beware the difference between linspace and logspace and that we're using only 100 points total
# phase plot in linear scale with 100pts will be off from logscale
if sys_und_tst.isdtime():
    w = 2 * np.pi * np.linspace(0, 0.5, nsamp_single)
else:
    w = 2 * np.pi * np.linspace(0, fstop_c, nsamp_single)

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

plt.show()
