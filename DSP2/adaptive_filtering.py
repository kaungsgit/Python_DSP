import matplotlib.pyplot as plt
import numpy as np
import custom_tools.fftplot as fftplot
import scipy.signal as sig
import scipy.fftpack as fft

# A/D Conversion, Sampling
nsamps = 2 ** 12
# generate time vector
fs = 500e6
ftone = 10e6
t = np.arange(nsamps) * 1 / fs

# adc_out = np.cos(2 * np.pi * ftone * t)

tone2 = 0
noise = 1 * np.random.randn(nsamps)

noise1 = 1 * np.random.randn(nsamps)
# adc_out = np.cos(2 * np.pi * ftone * t) + np.cos(2 * np.pi * tone2 * t)

s = np.cos(2 * np.pi * ftone * t)
adc_out = s + noise

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
x, y = fftplot.winfft(win_data, fs=fs)
plt.figure()
fftplot.plot_spectrum(x, y)
plt.title(f'Output Spectrum - {ftone / 1e6} MHz Tone')

w = np.zeros(nsamps + 1)
w[0] = 0.3
d = adc_out
x = noise1
y = np.zeros(nsamps)
e = np.zeros(nsamps)

plt.figure()
plt.subplot(5, 1, 1)
plt.plot(t, s)

plt.subplot(5, 1, 2)
plt.plot(t, x)

plt.subplot(5, 1, 3)
plt.plot(t, d)

plt.subplot(5, 1, 4)
plt.plot(t, e)

plt.subplot(5, 1, 5)
plt.plot(w)

for i in range(nsamps):
    y[i] = w[i] * x[i]
    e[i] = d[i] - y[i]
    w[i + 1] = w[i] + 0.005 * e[i] * x[i]

plt.figure()
plt.subplot(5, 1, 1)
plt.plot(t, s)

plt.subplot(5, 1, 2)
plt.plot(t, x)

plt.subplot(5, 1, 3)
plt.plot(t, d)

plt.subplot(5, 1, 4)
plt.plot(t, e)

plt.subplot(5, 1, 5)
plt.plot(w)

# using cusomized fft module
x, y = fftplot.winfft(e, fs=fs)
plt.figure()
fftplot.plot_spectrum(x, y)
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

plt.show()
