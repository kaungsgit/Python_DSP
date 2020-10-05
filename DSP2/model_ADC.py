import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import custom_tools.fftplot as fftplot
import custom_tools.handyfuncs as hf
import scipy.fftpack as fft
from matplotlib.ticker import StrMethodFormatter

Fs = 10e6
Ts = 1 / Fs
num_sampls = 10e3
x_t = np.arange(0, num_sampls * Ts, Ts)
f1 = 1e3
# inputv = 1001 * np.cos(np.linspace(0, 4 * np.pi, 2 ** 16))

inputv = 5*np.cos(2 * np.pi * f1 * x_t)

FS_p = 5
FS_n = -5
n = 4
steps = 2 ** n
q = (FS_p - FS_n) / steps
#
# for i in range(steps):
#     voltage = (i - 1) * (2) / (steps - 1)

plt.figure()
plt.plot(x_t, inputv)

fig, ax = plt.subplots()

# inputv = np.linspace(FS_n, FS_p, 1000)

# wrong equation, no need for (2 ** (n - 1)) * q
# y = np.clip((2 ** (n - 1)) * q * np.round(inputv / q), 0, steps - 1)

# y = np.clip(np.round(inputv / q), -steps, steps - 1)

y = np.clip(np.round(inputv / q), -2 ** (n - 1), 2 ** (n - 1) - 1)

format_str = '{' + 'x:0{}b'.format(n) + '}'

# ax.yaxis.set_major_formatter(StrMethodFormatter(format_str))

# plt.plot(inputv, 2 ** (n - 1) * inputv)
plt.plot(inputv, y)

# ytick_list = [i for i in range(2 ** n)]

ytick_list = [i for i in range(-2 ** (n - 1), 2 ** (n - 1))]

plt.yticks(ytick_list)


def ADC_model(input_list):
    sampled_data = 0
    for count, val in enumerate(input_list):
        if count % 2000 == 0:
            # sampled_data = (np.clip(np.round(val / q), 0, steps - 1))

            sampled_data = np.clip(np.round(val / q), -2 ** (n - 1), 2 ** (n - 1) - 1)

        yield sampled_data


adc_out_gen = ADC_model(inputv)

result = np.fromiter(adc_out_gen, float)

plt.figure()
plt.plot(inputv, label='Original Analog Waveform')
plt.plot(result, label='ADC Output Waveform')
plt.legend()
plt.title('ADC Output Waveform')

plt.show()
