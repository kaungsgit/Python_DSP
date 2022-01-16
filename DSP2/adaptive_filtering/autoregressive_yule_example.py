import matplotlib.pyplot as plt
import numpy as np
import custom_tools.fftplot as fftplot
import scipy.signal as sig
from scipy.linalg import toeplitz
import scipy.fftpack as fft
from numpy.random import randn
from spectrum import aryule
import pylab as pl

nsamps = 2 ** 12
noise = 2 * randn(nsamps)

b = [1]
a = [1, 1, 1, 0.4]

filtered_noise = sig.lfilter(b, a, noise)

corr_sig = sig.correlate(filtered_noise, filtered_noise)

corr_sig_0_end = corr_sig[filtered_noise.size - 1:]

plt.figure()
plt.plot(corr_sig_0_end)

corr_mat = toeplitz(corr_sig_0_end)

corr_vec = np.append(corr_sig_0_end[1:], 0)

a_vec = np.dot(np.linalg.inv(corr_mat), -corr_vec)
a_vec_r = np.append(1, a_vec[:200])

# y = sig.lfilter([1], a, pl.randn(1, nsamps))
ar, variance, coeff_reflection = aryule(filtered_noise, 20)

plt.figure()
plt.plot(a, label='original coeffs')
plt.plot(a_vec_r, label='estimated coeffs, manual')
plt.plot(np.append(1, ar), label='estimated coeffs, aryule')
plt.legend()
plt.show()

pass
