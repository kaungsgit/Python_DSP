import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import custom_tools.fftplot as fftplot
import custom_tools.handyfuncs as hf
import scipy.fftpack as fft
from matplotlib.ticker import StrMethodFormatter


def get_ytick(FS_n=0, FS_p=1, n=2):
    if FS_n >= 0:

        # Modifying ytick to be integer codes
        ytick_list = [i for i in range(2 ** n)]
    else:
        ytick_list = [i for i in range(-2 ** (n - 1), 2 ** (n - 1))]

    return ytick_list


def ADC_model(input_list, FS_n=0, FS_p=1, n=2, sample_mod=1):
    steps = 2 ** n
    q = (FS_p - FS_n) / steps

    sampled_data = 0

    if FS_n >= 0:

        for count, val in enumerate(input_list):
            if count % sample_mod == 0:
                sampled_data = (np.clip(np.round(val / q), 0, steps - 1))
            yield sampled_data

    else:

        for count, val in enumerate(input_list):
            if count % sample_mod == 0:
                sampled_data = np.clip(np.round(val / q), -2 ** (n - 1), 2 ** (n - 1) - 1)
            yield sampled_data


# analog wave creation
Fs = 10e6
Ts = 1 / Fs
num_sampls = 20e3
x_t = np.arange(0, num_sampls * Ts, Ts)
f1 = 1e3

# # wrong equation, no need for (2 ** (n - 1)) * q
# # y = np.clip((2 ** (n - 1)) * q * np.round(inputv / q), 0, steps - 1)

# formatting for binary y axis label, not really that useful, decimal is easier to read
# format_str = '{' + 'x:0{}b'.format(n) + '}'
# fig, ax = plt.subplots()
# # ax.yaxis.set_major_formatter(StrMethodFormatter(format_str))

''' ################################# Ramp input ######################################## '''
FS_n = 0
FS_p = 2
n = 2

# # ramp input for transfer function
inputv = np.linspace(0, 2, int(num_sampls))

plt.figure()
plt.plot(x_t * 1e3, inputv)
plt.title('Analog Waveform')
plt.xlabel('Time [ms]')
plt.ylabel('Voltage')

adc_out_gen_ramp = ADC_model(inputv, FS_n=FS_n, FS_p=FS_p, n=n, sample_mod=1)
ytick_list = get_ytick(FS_n=FS_n, FS_p=FS_p, n=n)

result = np.fromiter(adc_out_gen_ramp, float)

plt.figure()
plt.plot(inputv, result)

plt.xlabel('Voltage')
plt.ylabel('Code')
plt.title('Transfer Function of ADC')
plt.yticks(ytick_list)

''' ################################# cosine input ######################################## '''
FS_n = -5
FS_p = 5
n = 4
sample_mod = 100

# cosine wave analog input
inputv = 5 * np.cos(2 * np.pi * f1 * x_t)

plt.figure()
plt.plot(x_t * 1e3, inputv)
plt.title('Analog Waveform')
plt.xlabel('Time [ms]')
plt.ylabel('Voltage')

adc_out_gen_cos = ADC_model(inputv, FS_n=FS_n, FS_p=FS_p, n=n, sample_mod=sample_mod)
ytick_list = get_ytick(FS_n=FS_n, FS_p=FS_p, n=n)

result = np.fromiter(adc_out_gen_cos, float)

plt.figure()
plt.plot(result)
plt.xlabel('Sample')
plt.ylabel('Code/Voltage')
plt.title('ADC Output. Sampled at every {}'.format(sample_mod))
# plt.yticks(ytick_list)

plt.show()
