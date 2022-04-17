import matplotlib.pyplot as plt
import numpy as np
import custom_tools.fftplot as fftplot
import scipy.signal as sig
import scipy.fftpack as fft
from numpy.random import randn

# A/D Conversion, Sampling
num_samps = 2 ** 12
# generate time vector
fs = 500e6
ftone = 10e6
t = np.arange(num_samps) * 1 / fs

tone2 = 20e6
noise = 3 * randn(num_samps)

# noise[0:int(nsamps / 2)] = np.cos(2 * np.pi * tone2 * t[0:int(nsamps / 2)])

# noise = np.cos(2 * np.pi * tone2 * t)

noise1 = 1 * randn(num_samps)

s = 1 * np.cos(2 * np.pi * ftone * t)

delayed_noise = sig.lfilter([0, 0.2, 0.4, 0.6, 0.8, 1], 1, noise)

adc_out = s + delayed_noise

# using cusomized fft module
x, y = fftplot.winfft(adc_out, fs=fs)
plt.figure()
fftplot.plot_spectrum(x, y)
plt.title(f'Output Spectrum signal+delayed_noise - {ftone / 1e6} MHz Tone')

plt.figure()
plt.plot(t, adc_out)
plt.title(f'Time domain signal+delayed_noise - {ftone / 1e6} MHz Tone')

# windowing with kaiser
win_len = num_samps
beta = 12
win = sig.kaiser(win_len, beta)
winscale = np.sum(win)

win_data = win * adc_out

plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(win_data, fs=fs))
plt.title(f'Spectrum windowed signal+delayed_noise - {ftone / 1e6} MHz Tone')

plt.figure()
plt.plot(t, win_data)
plt.title(f'Time domain windowed signal+delayed_noise - {ftone / 1e6} MHz Tone')

# using cusomized fft module
# x, y = fftplot.winfft(win_data, fs=fs)


num_taps = 20
w0 = np.zeros(num_samps + 1)
w1 = np.zeros(num_samps + 1)

w = np.zeros([num_taps, num_samps + 1])
d = adc_out
x = noise
y = np.zeros(num_samps)
e = np.zeros(num_samps)

plt.figure()
plt.subplot(5, 1, 1)
plt.plot(t, s)
plt.ylabel('s')
plt.title(f's x d e w pre-loop - {ftone / 1e6} MHz Tone')

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

mu = 0.0001
for n in range(num_samps):

    convo_sum = 0
    for k in range(num_taps):
        convo_sum = convo_sum + w[k][n] * x[n - k]

    y[n] = convo_sum

    e[n] = d[n] - y[n]

    for k in range(num_taps):
        w[k][n + 1] = w[k][n] + 2 * mu * e[n] * x[n - k]

    # # one tap filter
    # y[i] = w[i] * x[i]
    # e[i] = d[i] - y[i]
    # w[i + 1] = w[i] + 0.005 * e[i] * x[i]

plt.figure()
plt.subplot(6, 1, 1)
plt.plot(t, s)
plt.ylabel('s')
plt.title(f's x d e w after running LMS - {ftone / 1e6} MHz Tone')

plt.subplot(6, 1, 2)
plt.plot(t, x)
plt.ylabel('x')

plt.subplot(6, 1, 3)
plt.plot(t, d)
plt.ylabel('d')

plt.subplot(6, 1, 4)
plt.plot(t, e)
plt.ylabel('e')

plt.subplot(6, 1, 5)
plt.ylabel('w')
# plt.plot(np.transpose(w))
for k in range(num_taps):
    # plt.legend(f'{k}')
    plt.plot(w[k], label=f'w{k}')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.subplot(6, 1, 6)
plt.plot(t, s - e)
plt.ylabel('s - e')

# using cusomized fft module
plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(e, fs=fs))
plt.title(f'Spectrum of error signal, aka output signal with noise cancelled out - {ftone / 1e6} MHz Tone')
#
# # example from book Digital Signal Processing (Third Edition) Fundamentals and Applications
# # table 9.1
#
# d = [-0.2947, 1.0017, 2.5827, -1.6019, 0.5622, 0.4456, -4.2674, -0.8418, -0.3862, 1.2274, 0.6021, 1.1647, 0.963,
#      -1.5065, -0.1329, 0.8146]
# x = [-0.5893, 0.5893, 3.1654, -4.6179, 1.1244, 2.3054, -6.5348, -0.2694, -0.7724, 1.0406, -0.7958, 0.9152, 1.926,
#      -1.5988, 1.7342, 3.0434]
# s = [0, 0.7071, 1, 0.7071, 0, -0.7071, -1, -0.7071, 0, 0.7071, 1, 0.7071, 0, -0.7071, -1, -0.7071]
# lenSig = len(d)
# w = np.zeros(lenSig + 1)
# w[0] = 0.3
# y = np.zeros(lenSig)
# e = np.zeros(lenSig)
#
# for i in range(lenSig):
#     y[i] = w[i] * x[i]
#     e[i] = d[i] - y[i]
#     w[i + 1] = w[i] + 0.01 * e[i] * x[i]
#
# plt.figure()
# plt.subplot(5, 1, 1)
# plt.plot(s)
#
# plt.subplot(5, 1, 2)
# plt.plot(x)
#
# plt.subplot(5, 1, 3)
# plt.plot(d)
#
# plt.subplot(5, 1, 4)
# plt.plot(e)
#
# plt.subplot(5, 1, 5)
# plt.plot(w)
#
# # autocorrelation (correlation vs delay for same function)
#
# # noise signal
# N = nsamps
# lag = np.arange(-N + 1, N)
# # noise = randn(N) + 1j * randn(N)
#
# corr = sig.correlate(noise, noise)
# sd_n = np.std(noise)
# m_n = np.mean(noise)
#
# plt.figure()
# plt.plot(lag, np.abs(corr))
# plt.xlabel("Lag [samples]")
# plt.ylabel("Real Magnitude")
# plt.title(f"AWGN Autocorrelation {N} Samples")
#
plt.show()
