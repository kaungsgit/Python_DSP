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
print('Using equation GCL_SetLev_Pout = GF/(1+GOL) aka GCL_SetLev_Pout = GF/(1+GF*GFB) with K={}'.format(K))
# looking at McNeill's notes, you can see GOL should be better termed as Loop Gain
# Forward Gain GF (from input to output)
# Feedback Gain GFB (from output to - term of sum node)
# Loop Gain aka Open Loop Gain GOL (break loop at - term, GF*GFB)
print(unreduced_Gcl)
reduced_sys = con.minreal((K * GF) / (1 + K * GF * GFB))
print(
    'Reduced GCL_SetLev_Pout. It should be the same as GCL_SetLev_Pout returned from con.feedback(K * GF, GFB) with K={}'.format(
        K))
print(reduced_sys)

n = np.arange(25)
# step is done on the closed loop system

plt.figure()
GCL_SetLev_Pout = con.feedback(K * GF, GFB)
# step respoonse for when Set Level is increased from 0 to 1500, GCL_SetLev_Pout => Set Level to Pout
step_size = 1500  # bits
n, yout = con.step_response(step_size * GCL_SetLev_Pout, T=n)
print('GCL_SetLev_Pout from con.feedback with K={}'.format(K))
print(GCL_SetLev_Pout)
plt.step(n, (np.squeeze(yout)), label=f"K = {K:0.2f}")

K = 0.25
GCL_SetLev_Pout = con.feedback(K * GF, GFB)
# step respoonse for when Set Level is increased from 0 to 1500, GCL_SetLev_Pout => Set Level to Pout
n, yout = con.step_response(step_size * GCL_SetLev_Pout, T=n)
plt.step(n, (np.squeeze(yout)), 'r', label=f"K = {K:0.2f}")

plt.legend()
plt.title("Power Control Loop Step Response\n Set Level changed from 0 to 1500 bits with Pin constant")
plt.ylabel("Power Gain [dB]")
plt.xlabel("Sample Number")

plt.figure()
K = 0.5
GOL_SetLev_Pout = K * GF * GFB
# forward gain is 1 from pin to pout, feedback gain is the same as GOL (from set level to pout)
GCL_Pin_Pout = con.feedback(1, GOL_SetLev_Pout)
# step respoonse for when Pin increases by 10dB with Set Level fixed, from Pin to Pout
# control loop adjust accordingly to ensure pout is same as set level
step_size = 10  # dB
n, yout = con.step_response(step_size * GCL_Pin_Pout, T=n)

# sys2 = con.tf(0.83155, [1, -1, 0], 1)
#
# n, yout = con.step_response(step_size * con.feedback(1, K * sys2), T=n)

print('GCL_SetLev_Pout from con.feedback with K={}'.format(K))
print(GCL_Pin_Pout)

plt.step(n, (np.squeeze(yout)), label=f"K = {K:0.2f}")

K = 0.25
GOL_SetLev_Pout = K * GF * GFB
GCL_Pin_Pout = con.feedback(1, GOL_SetLev_Pout)
# step respoonse for when Pin increases by 10dB with Set Level fixed, from Pin to Pout
n, yout = con.step_response(step_size * GCL_Pin_Pout, T=n)
plt.step(n, (np.squeeze(yout)), 'r', label=f"K = {K:0.2f}")

plt.legend()
plt.title("Power Control Loop Step Response\n Pin changed by 10dB with Set Level fixed")
plt.ylabel("Power Gain [dB]")
plt.xlabel("Sample Number")

plt.show()
