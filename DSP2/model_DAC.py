import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import custom_tools.fftplot as fftplot
import custom_tools.handyfuncs as hf
import scipy.fftpack as fft


def ds_list(input_values, input_size):
    input_len = len(input_values)

    # initialization
    sum1 = 0
    sum2 = 0
    out_array = np.zeros(input_len)

    # create output by iterating through input_values
    for count, input_ in enumerate(input_values):
        input_ = int(input_)

        # compute next state (clock update)
        sum1d = sum1
        sum2d = sum2

        # asynchronous operations
        out = -1 if sum2d < 0 else 1  # np.sign() returns 0 if 0
        fb = 2 ** (input_size - 1) * out
        fbx2 = 2 * fb
        delta1 = input_ - fb
        delta2 = sum1d - fbx2

        sum1 = sum1d + delta1
        sum2 = sum2d + delta2

        out_array[count] = out
        # end for

    return out_array


# Same model as ds_list returning a generator iterator instead
def ds_gen(input_values, input_size):
    # initialization
    sum1 = 0
    sum2 = 0

    # create output by iterating through input_values
    for input_ in input_values:
        input_ = int(input_)

        # compute next state (clock update)
        sum1d = sum1
        sum2d = sum2

        # asynchronous operations
        out = -1 if sum2d < 0 else 1  # np.sign() returns 0 if 0
        fb = 2 ** (input_size - 1) * out
        fbx2 = 2 * fb
        delta1 = input_ - fb
        delta2 = sum1d - fbx2

        sum1 = sum1d + delta1
        sum2 = sum2d + delta2

        yield out


def three_tap_moving_avg_list(input_values, input_size, coeffs=[1, 1, 1]):
    input_len = len(input_values)

    # initialization
    x = 0
    x1d = 0

    out_array = np.zeros(input_len)

    # # two tap FIR filter model
    # https://drive.google.com/file/d/1nyc_faIAnee3s5ZTRZZea4H2sNmLw-o-/view?usp=sharing
    # a = 0
    # b = 0
    # c = 0
    # d = 0
    # e = 0
    #
    # for count, input_ in enumerate(input_values):
    #     input_ = int(input_)
    #
    #     # synchronous operations - what happens on a risng clock edge?
    #     b = a
    #     # asynchronous operations - what happens in between clock edges?
    #     a = input_
    #     c = coeffs[0] * a
    #     e = coeffs[1] * b
    #     d = c + e
    #     out = d
    #
    #     out_array[count] = out

    # three tap FIR filter
    # https://drive.google.com/file/d/1wZlcKHbF2qm7BggPAS0U6PTVm6uJTwfV/view?usp=sharing
    # create output by iterating through input_values
    for count, input_ in enumerate(input_values):
        input_ = int(input_)

        # # synchronous operations - what happens on a risng clock edge?
        # b = a
        # # asynchronous operations - what happens in between clock edges?
        # a = input_
        # c = coeffs[0] * a
        # e = coeffs[1] * b
        # d = c + e
        # out = d

        # compute next state (clock update)
        # always assign the output first in a flip flop chain to mimic intermediate values propagating through
        x2d = x1d
        x1d = x

        # asynchronous operations
        x = input_
        yp = coeffs[0] * x + coeffs[1] * x1d
        out = yp + coeffs[2] * x2d

        out_array[count] = out
        # end for

    return out_array


def three_tap_moving_avg_gen(input_values, input_size, coeffs=[1, 1, 1]):
    # input_len = len(input_values)

    # initialization
    x = 0
    x1d = 0

    # out_array = np.zeros(input_len)

    # # two tap FIR filter model
    # https://drive.google.com/file/d/1nyc_faIAnee3s5ZTRZZea4H2sNmLw-o-/view?usp=sharing
    # a = 0
    # b = 0
    # c = 0
    # d = 0
    # e = 0
    #
    # for count, input_ in enumerate(input_values):
    #     input_ = int(input_)
    #
    #     # synchronous operations - what happens on a risng clock edge?
    #     b = a
    #     # asynchronous operations - what happens in between clock edges?
    #     a = input_
    #     c = coeffs[0] * a
    #     e = coeffs[1] * b
    #     d = c + e
    #     out = d
    #
    #     out_array[count] = out

    # three tap FIR filter
    # https://drive.google.com/file/d/1wZlcKHbF2qm7BggPAS0U6PTVm6uJTwfV/view?usp=sharing
    # create output by iterating through input_values
    for count, input_ in enumerate(input_values):
        input_ = int(input_)

        # # synchronous operations - what happens on a risng clock edge?
        # b = a
        # # asynchronous operations - what happens in between clock edges?
        # a = input_
        # c = coeffs[0] * a
        # e = coeffs[1] * b
        # d = c + e
        # out = d

        # compute next state (clock update)
        # always assign the output first in a flip flop chain to mimic intermediate values propagating through
        x2d = x1d
        x1d = x

        # asynchronous operations
        x = input_
        yp = coeffs[0] * x + coeffs[1] * x1d
        out = yp + coeffs[2] * x2d

        # out_array[count] = out
        # end for

        yield out

    # return out_array


# simple function defintion to modify a list
def double(in_list):
    out = []
    for item in in_list:
        if item == 5:
            print("Five detected!")
            out.append(500)
        else:
            out.append(5 * item)
    return out


# same function modified to be a generator function
def double_gen(in_list):
    # I added this to demonstrate how yield works
    yield ("You just called next() for the first time!")

    for item in in_list:
        if item == 5:
            print("Five detected!")
            yield 500
        else:
            yield 5 * item


if __name__ == '__main__':
    Fs = 1e6
    Ts = 1 / Fs
    num_sampls = 2 ** 16
    x_t = np.arange(0, num_sampls * Ts, Ts)

    f1 = 0.25e3
    # inputv = 1001 * np.cos(np.linspace(0, 4 * np.pi, 2 ** 16))
    inputv = 1001 * np.cos(2 * np.pi * f1 * x_t)

    plt.figure()
    plt.plot(inputv)
    plt.title('Input test signal')
    out = ds_list(inputv, input_size=12)

    plt.figure()
    plt.plot(out)
    plt.title('DAC Output')

    # Simple Moving Average low pass filter
    ntaps = 1000
    coeffs = np.ones(ntaps)
    filt_out = sig.lfilter(coeffs, ntaps, out)
    plt.plot(filt_out)
    # exponential MAF
    alpha = 0.999
    filt_out2 = sig.lfilter([1 - alpha], [1, -alpha], out)
    plt.plot(filt_out2)

    # frequency response of simple moving avg
    w, h = sig.freqz(coeffs, whole=True, fs=Fs)
    plt.figure()
    db_mag = hf.db(h)
    plt.plot((w - Fs / 2) / 1e3, fft.fftshift(db_mag), label='simple moving avg')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Filter Frequency Response')
    # both mag and phase plot
    # hf.responsePlot(w, h, 'simple moving avg frequency response')

    # frequency response of simple moving avg
    w, h = sig.freqz([1 - alpha], [1, -alpha], whole=True, fs=Fs)
    # plt.figure()
    db_mag = hf.db(h)
    plt.plot((w - Fs / 2) / 1e3, fft.fftshift(db_mag), label='exponential MAF')
    plt.legend()

    # example spectrum
    plt.figure()
    # using cusomized fft module imported earlier
    x, y = fftplot.winfft(out, fs=Fs, beta=12)
    fftplot.plot_spectrum(x, y)
    plt.title('Output Spectrum (Unfiltered)')

    plt.figure()
    x, y = fftplot.winfft(filt_out, fs=Fs, beta=12)
    fftplot.plot_spectrum(x, y)
    plt.title('Output Spectrum (Filtered Simp Mov Avg)')

    plt.figure()
    x, y = fftplot.winfft(filt_out2, fs=Fs, beta=12)
    fftplot.plot_spectrum(x, y)
    plt.title('Output Spectrum (Filtered Exp Mov Avg)')

    plt.figure()
    filt_model_out = three_tap_moving_avg_list(inputv, 12, coeffs=[1, 1, 1])
    plt.plot(filt_model_out)
    plt.title('Filtered Input with 3 tap FIR filter Model')

    plt.figure()
    plt.subplot(121)
    x, y = fftplot.winfft(inputv, fs=Fs, beta=12)
    fftplot.plot_spectrum(x, y, ceil=60)
    plt.title('Input inputv')

    plt.subplot(122)
    x, y = fftplot.winfft(filt_model_out, fs=Fs, beta=12)
    fftplot.plot_spectrum(x, y, ceil=60)
    plt.title('Filtered Input with 3 tap FIR filter Model')

    plt.show()
