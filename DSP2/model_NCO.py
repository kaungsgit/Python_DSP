# it is not necessary to include the import items below that have already been done
# but doing so for the convenience of moving this code block elsewhere (it contains its dependencies)
#
# Any that were already imported will be ignored, so no harm either.

import math  # math.cos() is 8x faster than np.cos() for a scalar
import collections
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import custom_tools.fftplot as fftplot
import custom_tools.handyfuncs as hf
import scipy.fftpack as fft


def nco(fcw, pcw=0, acc_size=28, lut_in_size=14, lut_out_size=12, nsamp=None):
    acc = pcw

    # this will convert input_values to a iterator if a scalar or list-like
    # input. If the input is already a iterator then this block is ignored.
    if not isinstance(fcw, collections.Iterator):
        # input is not a generator or iterator:
        if np.size(fcw) == 1:
            # input is a scalarA
            fcw = it.repeat(fcw)
        else:
            # convert array to iterator
            fcw = iter(fcw)

    for count, samp in enumerate(fcw):
        if nsamp:
            if count > nsamp: break
        acc = (samp + acc) % 2 ** acc_size
        lut_in = acc // 2 ** (acc_size - lut_in_size)
        angle = lut_in / 2 ** lut_in_size * 2 * math.pi
        yield round(2 ** (lut_out_size - 1) * math.cos(angle))


if __name__ == '__main__':
    # test NCO, single value FCW

    fout = 5.03e3
    fclk = 10e6
    acc_s = 32

    fres = fclk / 2 ** acc_s

    fcw = round(fout / fres)

    nco_gen = nco(fcw, acc_size=acc_s, nsamp=2 ** 15)

    # nco_gen is a generator, so cannot use np.array(nco_gen) directly
    # np.array(list(nco_gen)) also works but is not as memory efficient
    result = np.fromiter(nco_gen, float)

    plt.figure()
    plt.plot(result)
    plt.title('NCO Output Waveform')

    # frequency spectrum of NCO
    plt.figure()

    # using cusomized fft module imported earlier
    x, y = fftplot.winfft(result / max(result), fs=fclk, beta=12)
    fftplot.plot_spectrum(x, y)
    plt.title('NCO Output Spectrum')

    plt.show()
