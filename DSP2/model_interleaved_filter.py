import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import custom_tools.fftplot as fftplot
import custom_tools.handyfuncs as hf
import scipy.fftpack as fft

import DSP2.model_DAC as model_DAC
import DSP2.model_NCO as model_NCO

Fs = 10e6
Ts = 1 / Fs
num_sampls = 2 ** 16
x_t = np.arange(0, num_sampls * Ts, Ts)
f1 = 0.25e3
# inputv = 1001 * np.cos(np.linspace(0, 4 * np.pi, 2 ** 16))
inputv = 1001 * np.cos(2 * np.pi * f1 * x_t)

inputv = [2, 3, 4, 5, 0, 0, 0, 0]

data_gen_e = (y for idx, y in enumerate(inputv) if (idx % 2) == 0)
data_gen_o = (y for idx, y in enumerate(inputv) if (idx % 2) != 0)


def interleaved_filter(input_list):
    print(1)
    input_len = len(input_list)
    out_array_e = np.zeros(input_len)
    out_array_o = np.zeros(input_len)

    for idx, val in enumerate(input_list):
        print(idx)
        print(val)
        if (idx % 2) == 0:
            filter_out_e = model_DAC.three_tap_moving_avg_gen(next(data_gen), 12, coeffs=[1, 1, 1])
        else:
            filter_out_o = model_DAC.three_tap_moving_avg_gen(next(data_gen), 12, coeffs=[1, 1, 1])

        out_array_e[idx] = list(filter_out_e)

        out_array_o[idx] = list(filter_out_o)
        # yield filter_out1


# filter_out_IL = interleaved_filter(inputv)

filter_out_e = model_DAC.three_tap_moving_avg_gen(data_gen_e, 12, coeffs=[1, 1, 1])

filter_out_o = model_DAC.three_tap_moving_avg_gen(data_gen_o, 12, coeffs=[1, 1, 1])


for idx, y in enumerate(inputv):
    if (idx % 2) == 0:
        print(next(filter_out_e))
    else:
        print(next(filter_out_o))
pass
