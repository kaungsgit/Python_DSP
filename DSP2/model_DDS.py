import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import custom_tools.fftplot as fftplot
import custom_tools.handyfuncs as hf
import scipy.fftpack as fft

import DSP2.model_DAC as model_DAC
import DSP2.model_NCO as model_NCO

# Fs = 10e6
# Ts = 1 / Fs
# num_sampls = 2 ** 16
# x_t = np.arange(0, num_sampls * Ts, Ts)
# f1 = 0.25e3
# # inputv = 1001 * np.cos(np.linspace(0, 4 * np.pi, 2 ** 16))
# inputv = 1001 * np.cos(2 * np.pi * f1 * x_t)

fout = 5e3
fclk = 10e6
acc_s = 32

fres = fclk / 2 ** acc_s

fcw = round(fout / fres)

nco_gen = model_NCO.nco(fcw, acc_size=acc_s, nsamp=2 ** 15)

dds_gen = model_DAC.ds_gen(nco_gen, 12)

# nco_gen is a generator, so cannot use np.array(nco_gen) directly
# np.array(list(nco_gen)) also works but is not as memory efficient
result = np.fromiter(dds_gen, float)

plt.figure()
plt.plot(result)
plt.title('DDS Output Waveform')

# example spectrum
plt.figure()
# using cusomized fft module imported earlier
x, y = fftplot.winfft(result, fs=fclk, beta=12)
fftplot.plot_spectrum(x, y)
plt.title('DDS Output Spectrum (Unfiltered)')

plt.figure()
# Simple Moving Average low pass filter
ntaps = 1000
coeffs = np.ones(ntaps)
filt_out = sig.lfilter(coeffs, ntaps, result)
plt.plot(filt_out)

plt.show()
