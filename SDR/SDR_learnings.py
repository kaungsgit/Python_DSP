import komm
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import sys
import pprint as pp
import numpy.random as random

# sys.path.append("../")
import custom_tools.fftplot as fftplot
from scipy.fftpack import fft, fftshift, fftfreq, ifft


def db(x):
    # returns dB of number and avoids divide by 0 warnings
    x = np.array(x)
    x_safe = np.where(x == 0, 1e-7, x)
    return 20 * np.log10(np.abs(x_safe))


qam = komm.QAModulation(16)

print(f"Constellation: {qam.constellation}")
print(f"Symbol Mapping (Gray code): {qam.labeling}")

# create dictionary for mapping data words to symbols (matching what we created earlier above
# to demonstrate utility of komm library)
const_map = dict(zip(qam.labeling, qam.constellation))
print("\n")
print("Dictionary for Constellation Mapping for Symbols:")
pp.pprint(const_map)

sym_rate = 1e3

sym_length = 26  # duration of impulse response
oversamp = 4  # samples per symbol
alpha = 0.3  # Roll-off factor (set between 0 and 1)

pulse = komm.RaisedCosinePulse(rolloff=alpha, length_in_symbols=sym_length)
t = np.linspace(-sym_length / 2, sym_length / 2, sym_length * oversamp)
plt.figure()
plt.show()
# the function returns a non-causal impulse response, so added indexes to x axis
plt.plot(pulse.impulse_response(t))
plt.plot(pulse.impulse_response(t), 'o')
plt.title('RC Filter Impulse Response')
plt.xlabel('Samples')

# scaling by 4 to normalize impulse response for use as filter with 0 dB passband
coeff = pulse.impulse_response(t) / 4

w, h = sig.freqz(coeff, fs=sym_rate * oversamp)
plt.figure()
plt.plot(w, db(h))
plt.title('Raised Cosine Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid()

# example QAM modulated data

# create random data source
num_symb = 10

data = np.int8(16 * random.rand(num_symb))
print(f"Random data words: {data}")

print("\n")
symbols = [const_map[symbol] for symbol in data]
print("Mapped symbol locations in QAM-16 constellation")
print(symbols)

# Longer sequence with pulse shaping:
#
# create random data source
num_symb = 2 ** 14

data = np.int8(16 * random.rand(num_symb))

# create dictionary for mapping data to symbols
const_map = dict(zip(qam.labeling, qam.constellation))
symbols = [const_map[symbol] for symbol in data]

# To pulse shape the data, first we upsample the sequence by inserting zeros, and then
# pass that upsampled signal through the pulse shaping filter.

# For demonstration we will use the raised cosine filter, showing the eventual result
# after two root-raised cosine filters.

tx = np.zeros(len(data) * oversamp, dtype=complex)
tx[::oversamp] = symbols

# pass data through tx pulse shaping filter:

tx_shaped = sig.lfilter(coeff, 1, tx)

plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(tx_shaped, fs=sym_rate * oversamp), drange=120)

plt.title("Spectrum")
# plt.grid()

# eye diagram


num_cycles = 4  # number of symbols to display in eye diagram
windows = 200  # a window is one path across display shown

# resample data to at least 64x per symbol to emulate continuous waveform

resamp = int(np.ceil(64 / oversamp))
tx_resamp = sig.resample(tx_shaped, len(tx_shaped) * resamp)

samp_per_win = oversamp * resamp * num_cycles

# divide by number of samples per win and then
# pad zeros to next higher multiple using tx_eye = np.array(tx_shaped),
# tx_eye.resize(N)

N = len(tx_resamp) // samp_per_win

tx_eye = np.array(tx_resamp)
tx_eye.resize(N * samp_per_win)

grouped = np.reshape(tx_resamp, [N, samp_per_win])

transient = sym_length // 2
eye = np.real(grouped.T)

# create an xaxis in samples np.shape(eye) gives the
# 2 dimensional size of the eye data and the first element
# is the interpolated number of samples along the x axis
nsamps = np.shape(eye)[0]
xaxis = np.arange(nsamps) / resamp

plt.figure()

# plot showing continuous trajectory of
plt.plot(xaxis, eye[:, transient:transient + windows])

# actual sample locations
plt.plot(xaxis[::resamp], eye[:, transient:transient + windows][::resamp], 'b.')
plt.title("Eye Diagram")
plt.xlabel('Samples')
plt.grid()
plt.show()

plt.show()
