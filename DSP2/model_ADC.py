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
inputv = 1 + np.cos(2 * np.pi * f1 * x_t)

FS = 2
n = 2
steps = 2 ** n
q = FS / steps
#
# for i in range(steps):
#     voltage = (i - 1) * (2) / (steps - 1)

plt.figure()
plt.plot(x_t, inputv)

fig, ax = plt.subplots()

# inputv = np.linspace(0, FS, 1000)

y = np.clip((2 ** (n - 1)) * q * np.round(inputv / q), 0, steps - 1)

format_str = '{' + 'x:0{}b'.format(n) + '}'

ax.yaxis.set_major_formatter(StrMethodFormatter(format_str))

# plt.plot(inputv, 2 ** (n - 1) * inputv)
plt.plot(x_t, y)

ytick_list = [i for i in range(2 ** n)]

plt.yticks(ytick_list)


def ADC_model(input_list):
    sampled_data = 0
    for count, val in enumerate(input_list):
        if count % 500 == 0:
            sampled_data = (np.clip((2 ** (n - 1)) * q * np.round(val / q), 0, steps - 1))

        yield sampled_data


adc_out_gen = ADC_model(inputv)

result = np.fromiter(adc_out_gen, float)

plt.figure()
plt.plot(inputv, label='Original Waveform')
plt.plot(result, label='IL Filtered Waveform')
plt.legend()
plt.title('IL Filter Output Waveform')

plt.show()
