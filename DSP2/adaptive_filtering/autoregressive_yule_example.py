import matplotlib.pyplot as plt
import numpy as np
import custom_tools.fftplot as fftplot
import scipy.signal as sig
from scipy.linalg import toeplitz
import scipy.fftpack as fft
from numpy.random import randn
# from spectrum import aryule #@todo: issues with installing this spectrum module...
import pylab as pl

# https://mpastell.com/pweave/_downloads/AR_yw.html (something similar)

nsamps = 2 ** 12
noise = 2 * randn(nsamps)

# ARMA is exactly like the generalized filter difference equation with both FIR and IIR
b = [1]  # FIR portion
a = [1, -2.2137, 2.9403, -2.1697, 0.9606]  # IIR portion
# a = [1, 0]
filtered_noise = sig.lfilter(b, a, noise)
autocorr_noise = sig.correlate(noise, noise)
autocorr_filtered_noise = sig.correlate(filtered_noise, filtered_noise)

autocorr_sig_pos_only = autocorr_filtered_noise[filtered_noise.size - 1:]

plt.figure()
plt.plot(np.linspace(-noise.size + 1, noise.size - 1, (noise.size - 1) * 2 + 1),
         autocorr_noise, label='autocorr of WGN')
plt.xlabel('Lag')
plt.ylabel('rnn power')
plt.legend()

plt.figure()
plt.plot(autocorr_filtered_noise)
plt.title('autocorr of filtered noise')

autocorr_mat = toeplitz(autocorr_sig_pos_only)
# autocorr_vec starts from rxx[1] to rxx[M], which is zero-padded
# since rxx[M] doesn't exist from original autocorrelation
autocorr_vec = np.append(autocorr_sig_pos_only[1:], 0)
# https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations#:~:text=a%20quantum%20computer-,Matrix%20algebra,-%5Bedit%5D
# https://www.geeksforgeeks.org/analysis-algorithms-big-o-analysis/
# inverting a 2**12 x 2**12 matrix is very taxing, O(n^3)
a_est = np.dot(np.linalg.inv(autocorr_mat), -autocorr_vec)
a_est_w_a0 = np.append(1, a_est[:20])

# y = sig.lfilter([1], a, pl.randn(1, nsamps))
# inverting big toeplitz matrix can be done with some smart algo
# https://www.youtube.com/watch?v=AOX1ifbRfBU&ab_channel=AdamKashlak
# ar, variance, coeff_reflection = aryule(filtered_noise, 20)

plt.figure()
plt.plot(a, label='original coeffs')
plt.plot(a_est_w_a0, label='estimated coeffs, manual, inverting full size toeplitz')
# plt.plot(np.append(1, ar), label='estimated coeffs, aryule')
plt.title('AR Coeffs Comparison')
plt.legend()
plt.show()

pass
