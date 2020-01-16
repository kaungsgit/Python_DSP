import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.fftpack as fft

import sys

sys.path.append("../")

import custom_tools.fftplot as fftplot
# import src.gps.gps as gps
from numpy.random import randn

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def s_plane_plot(sfunc, limits=[3, 3, 10], nsamp=500):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    sigma = np.linspace(-limits[0], limits[0], nsamp)
    omega = sigma.copy()

    sigma, omega = np.meshgrid(sigma, omega)

    s = sigma + 1j * omega

    surf = ax.plot_surface(sigma, omega, np.abs(sfunc(s)), cmap=cm.flag)
    ax.set_zlim(0, limits[2])
    plt.xlabel('$\sigma$')
    plt.ylabel('$j\omega$')
    fig.tight_layout()


s_plane_plot(lambda s: 1 / s, nsamp=50)


def X(s):
    return 1 / ((s + .2 + .5j) * (s + .2 - .5j))


s_plane_plot(X, limits=[1, 1, 4], nsamp=40)

# practical approach to plot the frequency response (when s = jw)

# for H(s) 1/((s + .2+.5j)*(s + .2-.5j))

# create a vector of the coefficients for the numerator and denominator polynomials of s:
num = [1]
# (convolving the coefficients is the same as multiplying the polynomials)
den = np.convolve([1, 0.2 + .5j], [1, 0.2 - .5j])

w, h = sig.freqs(num, den, worN=np.linspace(-10, 10, 500))
plt.figure()
plt.plot(w, np.abs(h))
plt.xlabel('Frequency [rad/sec]')
plt.ylabel('Magnitude')
plt.title('Frequency Response for $H(s)$')
plt.axis([-1, 1, 0, 6])
plt.grid()

# showing same response as a log log plot (typical for analog filters)
plt.figure()
plt.semilogx(w, 20 * np.log10(np.abs(h)))
plt.xlabel('Frequency [rad/sec]')
plt.ylabel('Magnitude')
plt.title('Frequency Response for $H(s)$')
plt.grid()

# digital freq response
numz = [1, 1]
denz = [2, 0]
w, h = sig.freqz(numz, denz, worN=np.linspace(-np.pi, np.pi, 100))
plt.figure()
plt.plot(w, np.abs(h))
plt.axis([-np.pi, np.pi, 0, 1.2])
plt.grid()
plt.xlabel('Frequency [rad/sample]')
plt.ylabel('Magnitude')
plt.title('Frequency Response for 2 point moving average')

# A/D Conversion, Sampling
nsamps = 2 ** 16
# generate time vector
fs = 1e9
ftone = 9e6
t = np.arange(nsamps) * 1 / fs

tone = np.cos(2 * np.pi * ftone * t)

# using cusomized fft module
x, y = fftplot.winfft(tone, fs=fs)
plt.figure()
fftplot.plot_spectrum(x, y)
plt.title('Output Spectrum (Analog) - 9 MHz Tone')
plt.axis([-500, 500, -100, 0])

# The "analog" signal was simulated with a higher sampling rate of 1 GHz
# we can therefore simulate the effects of sampling the analog signal with a
# 100 MHz sampling clock by taking every 10th sample and filling in zeros for
# the rest (to see the unfolded spectrum beyond the first Nyquist zone):

tone_dig = np.zeros(nsamps)

tone_dig[::10] = 10 * tone[::10]

x, y = fftplot.winfft(tone_dig, fs=fs)
plt.figure()
fftplot.plot_spectrum(x, y)
plt.title('Output Spectrum (Digital - Unfolded) fs = 100 MHz 9 MHz tone')
plt.axis([-500, 500, -100, 0])

# Undersampling
# Consider the same with a input signal at 109 MHz:
# generate time vector

ftone2 = 109e6
t = np.arange(nsamps) * 1/fs

tone2 = np.cos(2*np.pi*ftone2*t)

# using cusomized fft module
x,y = fftplot.winfft(tone2, fs = fs)
plt.figure()
fftplot.plot_spectrum(x,y);
plt.title('Output Spectrum (Analog) - 109 MHz Tone')
plt.axis([-500, 500, -100, 0])

# When the 109 MHz signal is sampled at 100 MHz, the output result is the same
# as the 9 MHz signal
tone_dig2 = np.zeros(nsamps)

tone_dig2[::10] = 10*tone2[::10]

x,y = fftplot.winfft(tone_dig2, fs = fs)
plt.figure()
fftplot.plot_spectrum(x,y);
plt.title('Output Spectrum (Digital - Unfolded) fs = 100 MHz 109 MHz tone')
plt.axis([-500, 500, -100, 0])

plt.show()
