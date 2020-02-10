import csv
import matplotlib.pyplot as plt
import numpy as np
import custom_tools.fftplot as fftplot
import scipy.signal as sig
import scipy.fftpack as fft

insert_loss = []
freq = []


# custom functions used
def db(x):
    # returns dB of number and avoids divide by 0 warnings
    x = np.array(x)
    x_safe = np.where(x == 0, 1e-7, x)
    return 20 * np.log10(np.abs(x_safe))


with open('./Data/insertion_loss_cable.csv') as f:
    csv_reader = csv.reader(f, delimiter=',')

    for row in csv_reader:
        # print(row)
        freq.append(float(row[0]))
        insert_loss.append(float(row[1]))

n = len(insert_loss)
freq = np.array(freq)
insert_loss = np.array(insert_loss)

plt.figure()
plt.plot(freq, insert_loss)
plt.xlabel('Freq (Hz)')
plt.ylabel('Amplitude (dB)')

freq_step = np.mean(np.gradient(freq))
print('freq_step is {}MHz'.format(freq_step / 1e6))
# need 25MHz freq_step, so 17/10*25 = 42.5
fs = 1 / freq_step
print('Fs is {}'.format(fs))

freq_conv = freq / 1e12
freq_step_conv = np.mean(np.gradient(freq_conv))
print('freq_step is {}MHz'.format(freq_step_conv))
# need 25MHz freq_step, so 17/10*25 = 42.5
fs = 1 / freq_step_conv
print('Fs is {}'.format(fs))

plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(insert_loss, fs=fs), drange=120)

interp = 17
insert_loss_1 = np.zeros(len(insert_loss) * interp)
insert_loss_1[::interp] = insert_loss

freq_step_post_interp = freq_step / interp
freq_interp = np.arange(0.3e6, 8.5e9 + freq_step_post_interp * 17, freq_step_post_interp)

plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(insert_loss_1, fs=fs), drange=120)

# interpolation filter
# design multiband filter
#
# The equiripple filter algorithm (sig.remez) was used to design a
# multiband filter, concentrating reject at just the image frequency locations.
# This provides for greater rejection of the images than a low pass filter
# with the same number of taps would provide.

# goal 60 dB rejection, pass main signal at +/- 650 KHz
# reject the following bands (real filter so only positive frequencies given)
# # 2.5 KHz +/- 650 Hz = 1850 KHz to 3150 KHz
# # 5 KHz - 650 Hz to 5 KHz = 4350 KHz to 5Hz
#
# # converting to normalized frequency: (1 = fs)
# # passband:
p0 = 0
p1 = 0.3e3 / fs

# reject band 1
b1_0 = (1.384e3 - 0.2e3) / fs
# b1_1 = (1.384e3 + 0.2e3) / fs
#
# # reject band 2
# b2_0 = (2 * 1.384e3 - 0.2e3) / fs
# b2_1 = (2 * 1.384e3 + 0.2e3) / fs
#
# # reject band 3
# b3_0 = (3 * 1.384e3 - 0.2e3) / fs
# b3_1 = (3 * 1.384e3 + 0.2e3) / fs
#
# # reject band 4
# b4_0 = (4 * 1.384e3 - 0.2e3) / fs
# b4_1 = (4 * 1.384e3 + 0.2e3) / fs
#
# # reject band 5
# b5_0 = (5 * 1.384e3 - 0.2e3) / fs
# b5_1 = (5 * 1.384e3 + 0.2e3) / fs
#
# # reject band 6
# b6_0 = (6 * 1.384e3 - 0.2e3) / fs
# b6_1 = (6 * 1.384e3 + 0.2e3) / fs
#
# # reject band 7
# b7_0 = (7 * 1.384e3 - 0.2e3) / fs
# b7_1 = (7 * 1.384e3 + 0.2e3) / fs
#
# # reject band 8
# b8_0 = (8 * 1.384e3 - 0.2e3) / fs
# b8_1 = (8 * 1.384e3 + 0.2e3) / fs

b1_1 = 0.5

# p0 = 0
# p1 = 650/10_000
#
# # reject band 1
# b1_0 = 1850/10_000
# b1_1 = 3150/10_000
#
# # reject band 2
# b2_0 = 4350/10_000
# b2_1 = 0.5

print(f"Fractional passband: {p0}, {p1} cylces/sample")
print(f"Fractional stop band 1: {b1_0}, {b1_1} cylces/sample")
# print(f"Fractional stop band 2: {b2_0}, {b2_1} cylces/sample")

delta_f = b1_0 - p1
n_taps = 80
print(f"Number of taps used {n_taps}")

# scale the coefficients by the interpolation rate to maintain same signal level:
# coeff = interp * sig.remez(n_taps, [p0, p1, b1_0, b1_1, b2_0, b2_1,
#                                     b3_0, b3_1, b4_0, b4_1,
#                                     b5_0, b5_1, b6_0, b6_1,
#                                     b7_0, b7_1, b8_0, b8_1], [1, 0, 0, 0, 0, 0, 0, 0, 0])

coeff = interp * sig.remez(n_taps, [p0, p1, b1_0, b1_1], [1, 0])

# frequency response of multiband filter
w, h = sig.freqz(coeff, whole=True, fs=fs)
plt.figure()

db_mag = db(h)
plt.plot(w - fs / 2, fft.fftshift(db_mag))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.title('Interpolation filter frequency response')

# overlay filter and signal to review performance

w, h = sig.freqz(coeff, whole=True, fs=fs)
plt.figure()
wsig, hsig = fftplot.winfft(insert_loss_1, fs=fs)

plt.plot(wsig, db(hsig))

# filter response is decreased 50 dB to align visually with filter:
plt.plot(w - fs / 2, fft.fftshift(db(h)) - 50)
plt.title("Signal and Filter Overlaid")
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')

# filter signal with zero-insert using interpolation fitler
insert_loss_filt = sig.lfilter(coeff, 1, insert_loss_1)

plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(insert_loss_filt, fs=fs), drange=150)
plt.title("Spectrum after Zero Insert 4x and Filtered")

plt.figure()
plt.plot(freq_interp[::interp], insert_loss, label="Original")
# plt.figure()
shift = 35
func = lambda x: None if (x is 0) else -x
plt.plot(freq_interp[0:func(shift)], insert_loss_filt[shift:], label = "Interpolated")
# plt.plot(freq_interp, insert_loss_1, '.', label="Zero Inserted")

# plt.axis([.23, .28, -1, 1])
plt.grid()

plt.xlabel("Time [s]")
plt.ylabel("Magnitude")
plt.title('Original and Interpolated Signal')
plt.legend()
plt.show()

plt.show()
