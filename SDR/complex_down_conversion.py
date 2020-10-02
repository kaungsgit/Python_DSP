import csv
import matplotlib.pyplot as plt
import numpy as np
import custom_tools.fftplot as fftplot
import scipy.signal as sig
import scipy.fftpack as fft

pi = np.pi

def angle2(x):
    fin_res = []

    for i in x:
        imag = i.imag
        real = i.real

        if real == 0 and isinstance(real, float):
            real = 0

        if imag == 0 and isinstance(real, float):
            imag = 0

        res = np.arctan2(imag, real)

        fin_res.append(res)

    return np.array(fin_res)


# A/D Conversion, Sampling
nsamps = 2 ** 16
# generate time vector
fs = 50e6
ftone = 10e6
t = np.arange(nsamps) * 1 / fs

tone2 = 0
adc_out = np.cos(2 * np.pi * ftone * t)

# adc_out = np.cos(2 * np.pi * ftone * t) + np.cos(2 * np.pi * tone2 * t)

# using cusomized fft module
x, y = fftplot.winfft(adc_out, fs=fs)
plt.figure()
fftplot.plot_spectrum(x, y)
plt.title(f'Output Spectrum of adc_out - {ftone / 1e6} MHz Tone')
# plt.axis([-500, 500, -100, 0])

nco_freq = 10e6
nco_cosine = np.cos(2 * np.pi * nco_freq * t)
nco_sine = np.sin(2 * np.pi * nco_freq * t)

i_post_mix = adc_out * nco_cosine
q_post_mix = adc_out * nco_sine

# using cusomized fft module
x, y = fftplot.winfft(i_post_mix, fs=fs)
plt.figure()
fftplot.plot_spectrum(x, y)
plt.title(f'Output Spectrum of i_post_mix - {ftone / 1e6} MHz Tone')

# using cusomized fft module
x, y = fftplot.winfft(q_post_mix, fs=fs)
plt.figure()
fftplot.plot_spectrum(x, y)
plt.title(f'Output Spectrum of q_post_mix - {ftone / 1e6} MHz Tone')

plt.figure()
yf = fft.fft(adc_out)
xf = fft.fftfreq(nsamps, 1 / fs)
xf = fft.fftshift(xf)
yf = fft.fftshift(yf)
plt.plot(xf / 1e3, np.abs(yf))

plt.figure()
plt.stem(xf / 1e3, angle2(np.round(yf, 1)) * 180 / pi, use_line_collection=True)

plt.show()
