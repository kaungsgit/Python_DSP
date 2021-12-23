# packages used

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import sys
import pprint as pp
import numpy.random as random

sys.path.append("../")
import custom_tools.fftplot as fftplot

import control as con
import control.matlab as ctrl
import custom_tools.handyfuncs as hf

# set the continuous sysc in either vector form or s form
# set the c2d_method
# returns impulse response, freq response, and pz map of both cont and discrete systems

z = con.tf('z')
zm1 = 1 / z
s = con.tf('s')

# Fs = 10 / np.pi
Fs = 1e4
print('Sampling rate is {}'.format(Fs))
T = 1 / Fs

# create continuous time transfer function (following presentation example in back-up slides)
# sysc = con.tf(1, [1, 2, 2, 0])
# sysc = con.tf(1, [1, 0, 1])

K = 1
# sysc = K * 1 / (s ** 3 + 2 * s ** 2 + 2 * s + 0)
# sysc = K * 1 / ((s + 1j) * (s - 1j))
sysc = s / (s + 1000)

# remove imag parts in coeffs of sysc.den and num
# imag 0j present if using symbolic expression and root_locus function stuck in complex warning:
# "Casting complex values to real discards the imaginary part"
# using squeeze will remove more brackets than necessary so just use [0][0]
sysc_den = sysc.den[0][0].real
sysc.den = [[sysc_den]]
sysc_num = sysc.num[0][0].real
sysc.num = [[sysc_num]]

# 2nd order system with wn and zeta
fn = 1
wn = 2 * np.pi * fn
zeta = 0.2
# sysc = wn ** 2 / (s ** 2 + 2 * zeta * wn * s + wn ** 2)


print(sysc)

c2d_method = 'zoh'
# method (string, optional) – Method to be applied,
# ‘zoh’ Zero-order hold on the inputs (default)
# ‘foh’ First-order hold, currently not implemented
# ‘impulse’ Impulse-invariant discretization, currently not implemented
# ‘tustin’ Bilinear (Tustin) approximation, only SISO ‘matched’ Matched pole-zero method, only SISO


# plot impulse response
tc, youtc = con.impulse_response(sysc)
# plt.figure()
# plt.plot(tc, youtc)
# plt.title("Impulse Response - Continuous System")
# plt.xlabel("Time [s]")
# plt.ylabel("Magnitude")
# plt.grid()

# plot step response
tc_s, youtc_s = con.step_response(sysc)
# plt.figure()
# plt.plot(tc_s, youtc_s)
# plt.title("Step Response - Continuous System")
# plt.xlabel("Time [s]")
# plt.ylabel("Magnitude")
# plt.grid()

# plot frequency response
w = 2 * np.pi * np.logspace(-3, np.log10(0.5 * Fs), 100)
magc, phasec, w = con.freqresp(sysc, w)
# freq response returns mag and phase as [[[mag]]], [[[phase]]]
# squeeze reduces this to a one dimensional array, optionally can use mag[0][0]
magc = np.squeeze(magc)
phasec = np.squeeze(phasec)
# plt.figure()
# plt.semilogx(w / (2 * np.pi), hf.db(magc))
# plt.grid()
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Magnitude [dB]')
# plt.title('Frequency Response - Continuous System')

#######################################################################################
# # doing impulse invariance the long way...
# # partial fraction expansion
# [r, p, k] = sig.residue([1], [1, 2, 2, 0])
# print([r, p, k])
# print('f is {}'.format(1 / T))
#
# # combine terms
# GF = con.tf([r[0], 0], [1, -np.exp(p[0] * T)], T)
# print(GF)
# GFB = con.tf([r[1], 0], [1, -np.exp(p[1] * T)], T)
# print(GFB)
# sys3 = con.tf([r[2], 0], [1, -np.exp(p[2] * T)], T)
# print(sys3)
#
# # combine systems
# sysd = (GF + GFB + sys3)
# print("Transfer Function for Discrete System (using Method of Impulse Invariance):")
# print(sysd)
#######################################################################################

# construct discrete system
sysd1 = con.c2d(sysc, T, method=c2d_method)
print('Using sample system')
print(1 / T * sysd1)
print(sysd1)
sysd = 1 / T * sysd1

# plot impulse response

# create time vector as multiples of the sampling time
# from 0 to 7 seconds to match analog impulse response
nsamps = 7 // T
td = np.arange(nsamps) * T

td, youtd = con.impulse_response(sysd, td)
youtd = np.squeeze(youtd)
# plt.figure()
# plt.plot(td, youtd)
# plt.plot(td, youtd, 'o', label='Sample Locations')
# plt.title("Impulse Response - Discrete System")
# plt.xlabel("Time [s]")
# plt.ylabel("Magnitude")
# plt.grid()
# plt.legend()

td_s, youtd_s = con.step_response(T * sysd, td)
youtd_s = np.squeeze(youtd_s)
# plt.figure()
# plt.plot(td, youtd_s)
# plt.plot(td, youtd_s, 'o', label='Sample Locations')
# plt.title("Step Response - Discrete System")
# plt.xlabel("Time [s]")
# plt.ylabel("Magnitude")
# plt.grid()
# plt.legend()

# plot frequency response
w = 2 * np.pi * np.logspace(-3, np.log10(0.5 * Fs), 100)
magd, phased, w = con.freqresp(T * sysd, w)

# freq response returns mag and phase as [[[mag]]], [[[phase]]]
# squeeze reduces this to a one dimensional array, optionally can use mag[0][0]
magd = np.squeeze(magd)
phased = np.squeeze(phased)
# plt.figure()
# plt.semilogx(w / (2 * np.pi), hf.db(magd))
# plt.grid()
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Magnitude [dB]')
# plt.title('Frequency Response - Discrete System')

# comparing continuous to discrete
plt.figure()
plt.plot(tc, youtc, label='Continous')
plt.plot(td, youtd, 'o', label='Discrete')
plt.title("Impulse Response - Comparison")
plt.xlabel("Time [s]")
plt.ylabel("Magnitude")
plt.grid()
plt.legend()

plt.figure()
plt.plot(tc_s, youtc_s, label='Continous')
plt.plot(td_s, youtd_s, 'o', label='Discrete')
plt.title("Step Response - Comparison")
plt.xlabel("Time [s]")
plt.ylabel("Magnitude")
plt.grid()
plt.legend()

plt.figure()
plt.subplot(2, 1, 1)
plt.semilogx(w / (2 * np.pi), hf.db(magc), label='Continous')
plt.semilogx(w / (2 * np.pi), hf.db(magd), '--', label='Discrete')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.title('Frequency Response - Comparison')
plt.legend()

plt.subplot(2, 1, 2)
plt.semilogx(w / (2 * np.pi), np.unwrap(phasec) * 180 / np.pi, label='Continous')
plt.semilogx(w / (2 * np.pi), np.unwrap(phased) * 180 / np.pi, '--', label='Discrete')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [Deg]')
# plt.title('Frequency Response - Comparison')
plt.legend()

plt.figure()
plt.subplot(2, 1, 1)
plt.semilogx(w, hf.db(magc), label='Continous')
plt.semilogx(w, hf.db(magd), '--', label='Discrete')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')
plt.title('Frequency Response - Comparison - rad/s')
plt.legend()

plt.subplot(2, 1, 2)
plt.semilogx(w, np.unwrap(phasec) * 180 / np.pi, label='Continous')
plt.semilogx(w, np.unwrap(phased) * 180 / np.pi, '--', label='Discrete')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Phase [Deg]')
# plt.title('Frequency Response - Comparison')
plt.legend()

con.pzmap(sysc, title='Cont System Laplace plane')

con.pzmap(sysd, title='Discrete System Z plane')
if sysd.isdtime():
    cir_phase = np.linspace(0, 2 * np.pi, 500)
    plt.plot(np.real(np.exp(1j * cir_phase)), np.imag(np.exp(1j * cir_phase)), 'r--')
    plt.axis('equal')

plt.figure()
real, imag, freq = con.nyquist_plot(sysc)
scale = K / 2 + 1
plt.axis([-scale, scale, -scale, scale])
plt.title('Nyquist plot discrete')

# plt.figure()
rlist, klist = con.root_locus(sysc, grid=True)
plt.title('Root Locus cont')

plt.figure()
real, imag, freq = con.nyquist_plot(T * sysd, omega=w)
scale = K / 2 + 1
plt.axis([-scale, scale, -scale, scale])
plt.title('Nyquist plot discrete')

# plt.figure()
rlist, klist = con.root_locus(T * sysd, xlim=(-2, 2), ylim=(-2, 2), grid=True)
plt.title('Root Locus discrete')
if sysd.isdtime():
    cir_phase = np.linspace(0, 2 * np.pi, 500)
    plt.plot(np.real(np.exp(1j * cir_phase)), np.imag(np.exp(1j * cir_phase)), 'r--')
    # plt.axis('equal')

plt.show()
