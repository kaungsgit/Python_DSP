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


def db(x):
    # returns dB of number and avoids divide by 0 warnings
    x = np.array(x)
    x_safe = np.where(x == 0, 1e-7, x)
    return 20 * np.log10(np.abs(x_safe))


Fs = 10 / np.pi
T = 1 / Fs

# create continuous time transfer function (following presentation example in back-up slides)
sysc = con.tf(1, [1, 2, 2, 0])

print(sysc)

# plot impulse response

tc, youtc = con.impulse_response(sysc)

plt.figure()
plt.plot(tc, youtc)
plt.title("Impulse Response - Continuous System")
plt.xlabel("Time [s]")
plt.ylabel("Magnitude")
plt.grid()

# plot frequency response

w = 2 * np.pi * np.logspace(-3, np.log10(0.5 * Fs), 100)
magc, phasec, w = con.freqresp(sysc, w)

# freq response returns mag and phase as [[[mag]]], [[[phase]]]
# squeeze reduces this to a one dimensional array, optionally can use mag[0][0]
magc = np.squeeze(magc)
phasec = np.squeeze(phasec)
plt.figure()
plt.semilogx(w / (2 * np.pi), db(magc))
plt.grid()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.title('Frequency Response - Continuous System')

# partial fraction expansion
[r, p, k] = sig.residue([1], [1, 2, 2, 0])

print([r, p, k])

# combine terms


print('f is {}'.format(1 / T))
sys1 = con.tf([r[0], 0], [1, -np.exp(p[0] * T)], T)
print(sys1)
sys2 = con.tf([r[1], 0], [1, -np.exp(p[1] * T)], T)
print(sys2)
sys3 = con.tf([r[2], 0], [1, -np.exp(p[2] * T)], T)
print(sys3)

# combine systems
sysd = (sys1 + sys2 + sys3)

print("Transfer Function for Discrete System (using Method of Impulse Invariance):")
print(sysd)

# plot impulse response

# create time vector as multiples of the sampling time
# from 0 to 7 seconds to match analog impulse response

nsamps = 7 // T

td = np.arange(nsamps) * T

td, youtd = con.impulse_response(sysd, td)
youtd = np.squeeze(youtd)
plt.figure()
plt.plot(td, youtd)
plt.plot(td, youtd, 'o', label='Sample Locations')
plt.title("Impulse Response - Discrete System")
plt.xlabel("Time [s]")
plt.ylabel("Magnitude")
plt.grid()
plt.legend()

# plot frequency response
w = 2 * np.pi * np.logspace(-3, np.log10(0.5 * Fs), 100)
magd, phased, w = con.freqresp(T * sysd, w)

# freq response returns mag and phase as [[[mag]]], [[[phase]]]
# squeeze reduces this to a one dimensional array, optionally can use mag[0][0]
magd = np.squeeze(magd)
phased = np.squeeze(phased)
plt.figure()
plt.semilogx(w / (2 * np.pi), db(magd))
plt.grid()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.title('Frequency Response - Discrete System')

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
plt.semilogx(w / (2 * np.pi), db(magc), label='Continous')
plt.semilogx(w / (2 * np.pi), db(magd), '--', label='Discrete')
plt.grid()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.title('Frequency Response - Comparison')
plt.legend()

# z = con.tf('z')
# zm1 = 1 / z
# # sys_uT = z / (z - 1)
# sys_uT = 1 + 1 / z
#
# # sys_uT = 1 + z ** -1 + z ** -2
#
# tc, youtc = con.impulse_response(sys_uT)
#
# plt.figure()
# plt.plot(tc, youtc, 'o')
# plt.title("Impulse Response - discrete System")
# plt.xlabel("Time [s]")
# plt.ylabel("Magnitude")
# plt.grid()
#
# # plot frequency response
# omega = 2 * np.pi * np.logspace(-3, .2, 100)
# magd, phased, omega = con.freqresp(T * sys_uT, omega)
#
# # freq response returns mag and phase as [[[mag]]], [[[phase]]]
# # squeeze reduces this to a one dimensional array, optionally can use mag[0][0]
# magd = np.squeeze(magd)
# phased = np.squeeze(phased)
# plt.figure()
# plt.semilogx(omega, db(magd))
# plt.grid()
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Magnitude [dB]')
# plt.title('Frequency Response - Discrete System')
#
# # plot frequency response
# omega = 2 * np.pi * np.linspace(0, 1, 100)
# magd, phased, omega = con.freqresp(T * sys_uT, omega)
# # freq response returns mag and phase as [[[mag]]], [[[phase]]]
# # squeeze reduces this to a one dimensional array, optionally can use mag[0][0]
# magd = np.squeeze(magd)
# phased = np.squeeze(phased)
# plt.figure()
# plt.plot(omega, db(magd))
# plt.grid()
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Magnitude [dB]')
# plt.title('Frequency Response - Discrete System - linear scale')
#
# con.pzmap(sys_uT)
# phase = np.linspace(0, 2 * np.pi, 500)
# plt.plot(np.real(np.exp(1j * phase)), np.imag(np.exp(1j * phase)), 'r--')
# plt.axis('equal')
#
# # using matlab like syntax, control.matlab
# yout, t_val = ctrl.impulse(sys_uT)
# plt.figure()
# plt.stem(t_val, yout, use_line_collection=True)
# plt.title('Impulse response using control.matlab')


plt.show()
