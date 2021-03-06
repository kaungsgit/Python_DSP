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


def responsePlot(w, h, title):
    plt.subplot(2, 1, 1)
    plt.semilogx(w / (2 * np.pi), 20 * np.log10(np.abs(h)))
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.title(title)
    plt.subplot(2, 1, 2)
    plt.semilogx(w / (2 * np.pi), np.unwrap(np.angle(h)) * 360 / (2 * np.pi))
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Angle [deg]')
    # plt.show()


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
deci = 10
insert_loss_zeros_int = np.zeros(len(insert_loss) * interp)
insert_loss_zeros_int[::interp] = insert_loss

freq_step_post_interp = freq_step / interp
freq_interp = np.arange(0.3e6, 8.5e9 + freq_step_post_interp * interp, freq_step_post_interp)

freq_step_post_deci = freq_step_post_interp * deci
freq_deci = np.arange(0.3e6, 8.5e9 + freq_step_post_deci / deci, freq_step_post_deci)

plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(insert_loss_zeros_int, fs=fs), drange=120)

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
wsig, hsig = fftplot.winfft(insert_loss_zeros_int, fs=fs)

plt.plot(wsig, db(hsig))

# filter response is decreased 50 dB to align visually with filter:
plt.plot(w - fs / 2, fft.fftshift(db(h)) - 50)
plt.title("Signal and Filter Overlaid")
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')

# filter signal with zero-insert using interpolation fitler
insert_loss_filt = sig.lfilter(coeff, 1, insert_loss_zeros_int)

plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(insert_loss_filt, fs=fs), drange=150)
plt.title("Spectrum after Zero Insert {}x and Filtered".format(interp))

plt.figure()
plt.plot(freq_interp[::interp], insert_loss, label="Original")
# plt.figure()
shift = 2 * interp + 1
func = lambda x: None if (x is 0) else -x
plt.plot(freq_interp[0:func(shift)], insert_loss_filt[shift:], label="Interpolated")
plt.plot(freq_interp, insert_loss_zeros_int, '.', label="Zero Inserted")

# plt.axis([.23, .28, -1, 1])
plt.grid()

plt.xlabel("Freq [Hz]")
plt.ylabel("Magnitude")
plt.title('Original and Interpolated Signal')
plt.legend()

insert_loss_zeros_deci = np.zeros(len(insert_loss_filt))
insert_loss_zeros_deci[::deci] = deci * insert_loss_filt[::deci]
plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(insert_loss_zeros_deci, fs=fs), drange=120)
plt.title("Spectrum - Band Limited Noise - Zero Insert")

insert_loss_deci = insert_loss_zeros_deci[::deci] / deci
plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(insert_loss_deci, fs=fs / deci), drange=120)
plt.title("Spectrum - Band Limited Noise - Decimated to fs = {} KHz".format(fs / deci))

plt.figure()
plt.plot(freq_interp[::interp], insert_loss, label="Original")
plt.plot(freq_interp, insert_loss_filt, label="Interpolated")
plt.plot(freq_interp[::deci], insert_loss_deci, '.', label="Decimated")
plt.grid()
plt.xlabel("Freq [Hz]")
plt.ylabel("Magnitude")
plt.title('Interpolated and Decimated Signal')
plt.legend()

# interpolation with CIC
# comb aka moving avg
numz_comb = [1, -1]
denz_comb = [1]
# worN = np.linspace(-np.pi, np.pi, 100)
w, h = sig.freqz(numz_comb, denz_comb, whole=True, fs=fs)
plt.figure()
plt.plot(w - fs / 2, fft.fftshift(db(h)))
# plt.axis([-np.pi, np.pi, 0, max(np.abs(h))])
plt.grid()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.title('Frequency Response for moving avg filter')

# sallenKey filter
numz = [0, 0.009055917006062704]
denz = [1, -1.80967484, 0.81873075]
# worN = np.linspace(-np.pi, np.pi, 100)
w, h = sig.freqz(numz, denz, whole=True, fs=2 * np.pi * 1e6, worN=np.logspace(2, 6, 1000))
plt.figure()
responsePlot(w, h, 'Digital Filter Freq Response from SallenKey Class 3')

# rect integrator
numz_integ = [0, 1]
denz_integ = [1, -1]

# # trap integrator
# numz_integ = [1, 1]
# denz_integ = [2, -2]

w, h = sig.freqz(numz_integ, denz_integ, whole=True, fs=fs)

plt.figure()
plt.plot(w - fs / 2, fft.fftshift(db(h)))
# plt.axis([-np.pi, np.pi, 0, max(np.abs(h))])
plt.grid()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.title('Frequency Response for rectangular integrator')

# filter orig signal using moving average
insert_loss_postComb = sig.lfilter(numz_comb, denz_comb, insert_loss)

plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(insert_loss_postComb, fs=fs), drange=150)
plt.title("Spectrum after comb".format(interp))

insert_loss_upSamp = np.zeros(len(insert_loss_postComb) * interp)
insert_loss_upSamp[::interp] = insert_loss_postComb

plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(insert_loss_upSamp, fs=fs), drange=150)
plt.title("Spectrum after up sampling by {}".format(interp))

insert_loss_postInteg = sig.lfilter(numz_integ, denz_integ, insert_loss_upSamp)

plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(insert_loss_postInteg, fs=fs), drange=150)
plt.title("Spectrum after integrator".format(interp))

# plt.figure()
# plt.plot(insert_loss_postInteg)
# plt.plot(insert_loss_filt)

plt.figure()
plt.plot(freq_interp[::interp], insert_loss, label="Original")
# plt.plot(freq_interp, insert_loss_filt, label="Interpolated")
plt.plot(freq_interp, insert_loss_postInteg, '.', label="CIC Interpolated")
plt.grid()
plt.xlabel("Freq [Hz]")
plt.ylabel("Magnitude")
plt.title('Interpolated and Decimated Signal')
plt.legend()

plt.show()
