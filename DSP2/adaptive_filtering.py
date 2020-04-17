import matplotlib.pyplot as plt
import numpy as np
import custom_tools.fftplot as fftplot
import scipy.signal as sig
import scipy.fftpack as fft
from numpy.random import randn

# A/D Conversion, Sampling
nsamps = 2 ** 12
# generate time vector
fs = 500e6
ftone = 10e6
t = np.arange(nsamps) * 1 / fs

tone2 = 0
noise = 1 * randn(nsamps)

noise1 = 1 * randn(nsamps)

s = np.cos(2 * np.pi * ftone * t)

delayed_noise = sig.lfilter([1, 2, 3, 4], 1, noise)

adc_out = s + delayed_noise

# using cusomized fft module
x, y = fftplot.winfft(adc_out, fs=fs)
plt.figure()
fftplot.plot_spectrum(x, y)
plt.title(f'Output Spectrum - {ftone / 1e6} MHz Tone')

plt.figure()
plt.plot(t, adc_out)

# windowing with kaiser
win_len = nsamps
beta = 12
win = sig.kaiser(win_len, beta)
winscale = np.sum(win)

win_data = win * adc_out

plt.figure()
plt.plot(t, win_data)

# using cusomized fft module
# x, y = fftplot.winfft(win_data, fs=fs)
plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(win_data, fs=fs))
plt.title(f'Output Spectrum - {ftone / 1e6} MHz Tone')

nTaps = 10
w0 = np.zeros(nsamps + 1)
w1 = np.zeros(nsamps + 1)

w = np.zeros([nTaps, nsamps + 1])
d = adc_out
x = noise
y = np.zeros(nsamps)
e = np.zeros(nsamps)

plt.figure()
plt.subplot(5, 1, 1)
plt.plot(t, s)
plt.ylabel('s')

plt.subplot(5, 1, 2)
plt.plot(t, x)
plt.ylabel('x')

plt.subplot(5, 1, 3)
plt.plot(t, d)
plt.ylabel('d')

plt.subplot(5, 1, 4)
plt.plot(t, e)
plt.ylabel('e')

plt.subplot(5, 1, 5)
plt.plot(np.transpose(w))
plt.ylabel('w')

mu = 0.005
for i in range(nsamps):

    sum1 = 0
    for k in range(nTaps):
        sum1 = sum1 + w[k][i] * x[i - k]

    y[i] = sum1

    e[i] = d[i] - y[i]

    for k in range(nTaps):
        w[k][i + 1] = w[k][i] + 2 * mu * e[i] * x[i - k]

    # # one tap filter
    # y[i] = w[i] * x[i]
    # e[i] = d[i] - y[i]
    # w[i + 1] = w[i] + 0.005 * e[i] * x[i]

plt.figure()
plt.subplot(5, 1, 1)
plt.plot(t, s)
plt.ylabel('s')

plt.subplot(5, 1, 2)
plt.plot(t, x)
plt.ylabel('x')

plt.subplot(5, 1, 3)
plt.plot(t, d)
plt.ylabel('d')

plt.subplot(5, 1, 4)
plt.plot(t, e)
plt.ylabel('e')

plt.subplot(5, 1, 5)
plt.ylabel('w')
# plt.plot(np.transpose(w))
for k in range(nTaps):
    # plt.legend(f'{k}')
    plt.plot(w[k], label=f'w{k}')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# using cusomized fft module
plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(e, fs=fs))
plt.title(f'Output Spectrum - {ftone / 1e6} MHz Tone')

# example from book Digital Signal Processing (Third Edition) Fundamentals and Applications
# table 9.1

d = [-0.2947, 1.0017, 2.5827, -1.6019, 0.5622, 0.4456, -4.2674, -0.8418, -0.3862, 1.2274, 0.6021, 1.1647, 0.963,
     -1.5065, -0.1329, 0.8146]
x = [-0.5893, 0.5893, 3.1654, -4.6179, 1.1244, 2.3054, -6.5348, -0.2694, -0.7724, 1.0406, -0.7958, 0.9152, 1.926,
     -1.5988, 1.7342, 3.0434]
s = [0, 0.7071, 1, 0.7071, 0, -0.7071, -1, -0.7071, 0, 0.7071, 1, 0.7071, 0, -0.7071, -1, -0.7071]
lenSig = len(d)
w = np.zeros(lenSig + 1)
w[0] = 0.3
y = np.zeros(lenSig)
e = np.zeros(lenSig)

for i in range(lenSig):
    y[i] = w[i] * x[i]
    e[i] = d[i] - y[i]
    w[i + 1] = w[i] + 0.01 * e[i] * x[i]

plt.figure()
plt.subplot(5, 1, 1)
plt.plot(s)

plt.subplot(5, 1, 2)
plt.plot(x)

plt.subplot(5, 1, 3)
plt.plot(d)

plt.subplot(5, 1, 4)
plt.plot(e)

plt.subplot(5, 1, 5)
plt.plot(w)

# autocorrelation (correlation vs delay for same function)

# noise signal
N = nsamps
lag = np.arange(-N + 1, N)
# noise = randn(N) + 1j * randn(N)

corr = sig.correlate(noise, noise)
sd_n = np.std(noise)
m_n = np.mean(noise)

plt.figure()
plt.plot(lag, np.abs(corr))
plt.xlabel("Lag [samples]")
plt.ylabel("Real Magnitude")
plt.title(f"AWGN Autocorrelation {N} Samples")

plt.show()
