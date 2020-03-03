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

K = 1
GOLz = con.tf(0.83155 * K, [1, -1, 0], 1)

plt.figure()
real, imag, freq = con.nyquist_plot(GOLz, omega=np.linspace(0, np.pi, 1000))
plt.title('Nyquist plot of GOL with K={}'.format(K))
plt.axis([-1.4, .5, -10, 10])

# Modified Nyquist Plot:
# A Nyquist Plot of -1/Gol will show the range of K for stability

plt.figure()
real, imag, freq = con.nyquist_plot(-1 / GOLz, omega=np.linspace(0, np.pi, 1000))
plt.title('Modified Nyquist plot of -1/GOL with K={} \nRange of stability for K is 0-1.2'.format(K))
plt.axis([-2.5, 1.5, -2.5, 2.5])

rlist, klist = con.root_locus(GOLz, xlim=(-3, 3), ylim=(-3, 3), grid=True, Plot=True)
cir_phase = np.linspace(0, 2 * np.pi, 500)
plt.plot(np.real(np.exp(1j * cir_phase)), np.imag(np.exp(1j * cir_phase)), 'r--')
plt.title('Root Locus using control.root_locus with K={}'.format(K))
# plt.axis('equal')
# fig.axes.set(xlim=(-3, 3), ylim=(-3, 3))

plt.figure()
ax1 = plt.subplot(1, 1, 1)
ax1.plot(np.real(rlist), np.imag(rlist))
plt.title('Root Locus normal plot with K={}'.format(K))
angle = np.linspace(0, 2 * np.pi, 512)
ax1.plot(np.real(np.exp(1j * angle)), np.imag(np.exp(1j * angle)))
plt.axis([-2, 2, -2, 2])

# Power Control Loop step response

GF = con.tf(0.01464, [1, -1], 1)
GFB = con.tf(56.8, [1, 0], 1)
print('GF aka Forward Gain (GF) without K')
print(GF)
print('GFB aka Feedback Gain (GFB)')
print(GFB)

K = 0.5

unreduced_Gcl = (K * GF) / (1 + K * GF * GFB)
print('Using equation GCL = GF/(1+GOL) aka GCL = GF/(1+GF*GFB) with K={}'.format(K))
# looking at McNeill's notes, you can see GOL should be better termed as Loop Gain
# Forward Gain GF (from input to output)
# Feedback Gain GFB (from output to - term of sum node)
# Loop Gain aka Open Loop Gain GOL (break loop at - term, GF*GFB)
print(unreduced_Gcl)
reduced_sys = con.minreal((K * GF) / (1 + K * GF * GFB))
print('Reduced GCL. It should be the same as GCL returned from con.feedback(K * GF, GFB) with K={}'.format(K))
print(reduced_sys)

n = np.arange(25)
# step is done on the closed loop system

plt.figure()
GCL = con.feedback(K * GF, GFB)
n, yout = con.step_response(1500 * GCL, T=n)
print('GCL from con.feedback with K={}'.format(K))
print(GCL)

plt.step(n, (np.squeeze(yout)), label=f"K = {K:0.2f}")

K = 0.25
GCL = con.feedback(K * GF, GFB)
n, yout = con.step_response(1500 * GCL, T=n)
plt.step(n, (np.squeeze(yout)), 'r', label=f"K = {K:0.2f}")

plt.legend()
plt.title("Power Control Loop Step Response")
plt.ylabel("Power Gain [dB]")
plt.xlabel("Sample Number")

plt.show()
