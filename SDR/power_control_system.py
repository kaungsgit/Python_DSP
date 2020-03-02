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
golz = con.tf(0.83155 * K, [1, -1, 0], 1)

plt.figure()
real, imag, freq = con.nyquist_plot(golz, omega=np.linspace(0, np.pi, 1000))
plt.axis([-1.4, .5, -10, 10])

# Modified Nyquist Plot:
# A Nyquist Plot of -1/Gol will show the range of K for stability

plt.figure()
real, imag, freq = con.nyquist_plot(-1 / golz, omega=np.linspace(0, np.pi, 1000))

plt.axis([-2.5, 1.5, -2.5, 2.5])

rlist, klist = con.root_locus(golz, Plot=False)
plt.figure()
ax1 = plt.subplot(1, 1, 1)
ax1.plot(np.real(rlist), np.imag(rlist))
plt.title('Root Locus')
angle = np.linspace(0, 2 * np.pi, 512)
ax1.plot(np.real(np.exp(1j * angle)), np.imag(np.exp(1j * angle)))
plt.axis([-2, 2, -2, 2])

# Power Control Loop step response

sys1 = con.tf(0.01464, [1, -1], 1)
sys2 = con.tf(56.8, [1, 0], 1)
print('sys1 aka Forward Gain (GF) without K')
print(sys1)
print('sys2 aka Feedback Gain (GFB)')
print(sys2)

K = 0.5

unreduced_sys = (K * sys1) / (1 + K * sys1 * sys2)
print('Using equation GCL = GF/(1+GOL) aka GCL = GF/(1+GF*GFB) with K={}'.format(K))
# looking at McNeill's notes, you can see GOL should be better termed as Loop Gain
# Forward Gain GF (from input to output)
# Feedback Gain GFB (from output to - term of sum node)
# Loop Gain aka Open Loop Gain (break loop at - term, GF*GFB)
print(unreduced_sys)
reduced_sys = con.minreal((K * sys1) / (1 + K * sys1 * sys2))
print('Reduced sys. It should be the same as GCL returned from con.feedback(K * sys1, sys2) with K={}'.format(K))
print(reduced_sys)

n = np.arange(25)
# step is done on the closed loop system

plt.figure()
GCL = con.feedback(K * sys1, sys2)
n, yout = con.step_response(1500 * GCL, T=n)
print('GCL from con.feedback with K={}'.format(K))
print(GCL)

plt.step(n, (np.squeeze(yout)), label=f"K = {K:0.2f}")

K = 0.25
GCL = con.feedback(K * sys1, sys2)
n, yout = con.step_response(1500 * GCL, T=n)
plt.step(n, (np.squeeze(yout)), 'r', label=f"K = {K:0.2f}")

plt.legend()
plt.title("Power Control Loop Step Response")
plt.ylabel("Power Gain [dB]")
plt.xlabel("Sample Number")

plt.show()
