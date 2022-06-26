import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy.random as random
import komm

import pprint as pp

# QAM waveform with phase offset

qam = komm.QAModulation(16)

# create dictionary for mapping data words to symbols (matching what we created earlier above
# to demonstrate utility of komm library)
const_map = dict(zip(qam.labeling, qam.constellation))
print("\n")
print("Dictionary for Constellation Mapping for Symbols:")
pp.pprint(const_map)

# We will demonstrate Timing Error Detection
# with an oversampled waveform to easily show the error vs timing offset


# first upsample the data
# with a raised cosine pulse shaping filter (see CLass 1):
num_symb = 24000
sym_rate = 1e3
sym_length = 26  # duration of impulse response
oversamp = 4  # samples per symbol
alpha = 0.5  # Roll-off factor (set between 0 and 1)
data = np.int8(16 * random.rand(num_symb))
symbols = [const_map[symbol] for symbol in data]

pulse = komm.RaisedCosinePulse(rolloff=alpha, length_in_symbols=sym_length)
index = np.linspace(-sym_length / 2, sym_length / 2, sym_length * oversamp)

# scaling by 4 to normalize impulse response for use as filter with 0 dB passband
coeff = pulse.impulse_response(index) / 4

tx = np.zeros(num_symb * oversamp, dtype=complex)
tx[::oversamp] = symbols

# pass data through tx pulse shaping filter:

tx_shaped = sig.lfilter(coeff, 1, tx)

# eye diagram


num_cycles = 4  # number of symbols to display in eye diagram
windows = 200  # a window is one path across display shown
upsample = 16
# resample data to 64x per symbol to emulate continuous waveform
tx_resamp = sig.resample(tx_shaped, len(tx_shaped) * upsample)

samp_per_win = oversamp * upsample * num_cycles

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
xaxis = np.arange(nsamps) / upsample

plt.figure()

# plot showing continuous trajectory of
plt.plot(xaxis, eye[:, transient:transient + windows])

# actual sample locations
plt.plot(xaxis[::upsample], eye[:, transient:transient + windows][::upsample], 'b.')
plt.title("Eye Diagram")
plt.xlabel('Samples')
plt.grid()
# plt.show()


# we will use tx-resamp which has 64 samples per symbol to demonstrate
# the results of the Garnder Timing Error Detector vs Time offset

# this is just to show the results for a static timing offset rather
# than what would be used in a dynamic loop which would operate
# sample by sample

def ted(tx, n, offset):
    '''
    tx: oversampled complex waveform
    n: oversampling rate
    offset: sample offset delay
    '''
    # downsample to 2 samples per symbol with timing offset
    tx2 = tx[offset::int(n / 2)]

    # generate a prompt, late and early each offset by 1 sample
    late = tx2[2:]
    early = tx2[:-2]
    prompt = tx2[1:-1]

    # compute and return the Garnder Error result
    return np.real(np.conj(prompt[::2]) * (late - early)[::2])


# generate all offsets over symbol duration
offsets = np.arange(upsample * oversamp)

# rotate offsets to center error detection plot:
offsets = np.roll(offsets, -25)

# increment through each offset and compute the average timing error
ted_results = [np.mean(ted(tx_resamp, upsample * oversamp, offset)) for offset in offsets]

plt.figure()
plt.plot(np.arange(64) - 31, ted_results)

plt.title('Gardner TED Results for QAM Waveform')
plt.xlabel('Sample Offset')
plt.ylabel('Measured Timing Error')
plt.grid(True)


def mandm(tx, n, offset):
    '''
    tx: oversampled waveform
    n: oversampling rate
    offset: sample offset delay
    '''
    # downsample to 1 sample per symbol with timing offset
    tx2 = tx[offset::n]

    # compute M&M on real axis
    sign_tx2_real = np.sign(np.real(tx2))
    mm_real = np.real(tx2[1:]) * sign_tx2_real[:-1] - np.real(tx2[:-1]) * sign_tx2_real[1:]
    # compute M&M on imag axis
    sign_tx2_imag = (np.sign(np.imag(tx2)))
    mm_imag = np.imag(tx2[1:]) * sign_tx2_imag[:-1] - np.imag(tx2[:-1]) * sign_tx2_imag[1:]

    return mm_real + mm_imag


# increment through each offset and compute the average timing error
mandm_results = [np.mean(mandm(tx_resamp, upsample * oversamp, offset)) for offset in offsets]

plt.figure()
plt.plot(np.arange(64) - 31, mandm_results)
plt.grid()

plt.title('M&M Results for QAM Waveform')
plt.xlabel('Sample Offset')
plt.ylabel('Measured Timing Error')
plt.grid(True)

plt.show()