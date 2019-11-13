"""
@author: Kaung Myat San Oo
Animation for understanding rotating phasors, negative frequency in DSP concepts

Add, remove, or modify signals in rotating_phasors
Script will animate the individual phasors and the combined output in both rectangular and polar plots
Press any keyboard key to pause the animation

"""

# @todo
# find out why the update is slow
# reorganize the object oriented scheme
# fix rect plot time axis

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import cmath as cm
from collections.abc import Iterable
import matplotlib.patheffects as pe
from scipy.fftpack import fft, fftshift, fftfreq, ifft
import math

pause = False

pi = np.pi
amp = 1  # 1V        (Amplitude)
f = 1000  # 1kHz      (Frequency)
fs = 200000  # 200kHz    (Sample Rate)
T = 1 / f
w = 2 * pi * f

Ts = 1 / fs
num_sampls = fs  # number of samples

x_t = np.arange(0, fs * Ts, Ts)
n = x_t

# Select if you want to display the sine as a continuous wave
#  True = continuous (not able to zoom in x-direction)
#  False = Non-continuous  (able to zoom)
continuous = False

# if set True, all phasors in rotating_phasors will spin with respect to center of polar plot
# if False, all phasors will spin with respect to the end of the previous phasor end point (true vector addition)
spin_orig_center = False

# list of phasor arrays that get passed into animation function
rotating_phasors = []

# pass in phasor arrays directly when True
# this flag must be set to False if you're working with input_vector and FT_mode
pass_direct_phasor_list = False

FT_mode = True
input_vector = [3, 2, 1]
N = len(input_vector)

if pass_direct_phasor_list:
    # manual phasor list
    phi = 0
    rotating_phasors = [
        # np.array(amp * np.exp(1j * (2 * pi * 0 * x_t + pi))),
        np.array(amp * 1 * np.exp(1j * (2 * pi * (f / 1.0) * x_t + phi))),
        np.array(amp * 1 * np.exp(1j * (2 * pi * (-f / 1.0) * x_t + phi)))
    ]

else:
    if FT_mode:
        # in this mode, input vector is time  domain samples (real or complex) and the polar plot will show reconstructed
        # time domain samples
        xn = np.array(input_vector)
        Xk = fft(xn)

        # inverse FFT to check, should be the same as input_vector
        inverseFFT = ifft(xn)
        # rewriting inverse DFT equation
        # xn = 1/N summation k=0 to N-1 {Xk * e^(j * 2pi * k * n / N)
        for k, X_curr_k in enumerate(Xk):
            rotating_phasors.append(1 / N * X_curr_k *
                                    np.array(np.exp(1j * (2 * pi * f * k * n / N))))

    else:
        # in this mode, input vector is FIR filter coefficients h[k] or bk
        # polar plot will show how frequency response of FIR filter is made
        filt_coeffs = np.array(input_vector)

        # rewriting frequency response of FIR filter
        # H(e^jw) = summation k=0 to M-1 {bk e^(-j * w * k)}
        for k, b_curr_k in enumerate(filt_coeffs):
            rotating_phasors.append(b_curr_k *
                                    np.array(np.exp(-1j * (w * k * n))))

num_sigs = len(rotating_phasors)


def wrap_around(x, count):
    return x % count


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


# flatten list items
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


class ScopeRectCmbd(object):
    def __init__(self, ax, num_sigs, legend_list, xylabels, max_t=2 * T, dt=Ts):
        self.ax_r = ax
        self.ax_r.grid(True)
        self.dt = dt
        self.max_t = max_t
        self.t_data = np.empty(0)
        self.y_data = []
        self.sig_lines_r = []
        self.legend_list = legend_list
        self.xylabels = xylabels

        # data lines for drawing the phasor, edge tracing, real and image projection for each item in rotating_phasors
        for _ in range(num_sigs):
            self.sig_lines_r.append([self.ax_r.plot([], [], fstr)[0] for _, fstr in zip(range(2), ['', ''])])
            self.y_data.append([[0], [0]])

        # data lines for drawing the phasor, edge tracing, real and image projection for combined output signals in
        # rotating_phasors
        self.sig_lines_r_cmbd = [self.ax_r.plot([], [], fstr, linewidth=lw,
                                                path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()])[0]
                                 for _, fstr, lw in
                                 zip(range(2), ['r-', 'b-'], [3, 2])]
        self.y_data_cmbd = [[], []]

        for line, legend in zip(self.sig_lines_r_cmbd, self.legend_list):
            line.set_label(legend)

        self.sig_lines_r_curr_pt = [self.ax_r.plot([], [], fstr, linewidth=lw,
                                                   path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()])[
                                        0]
                                    for _, fstr, lw in
                                    zip(range(2), ['rx', 'bx'], [3, 2])]
        self.y_data_curr_pt = [0, 0]

        # # adding legend
        # self.sig_lines_r_cmbd[0].set_label('In-phase or Real')
        # self.sig_lines_r_cmbd[1].set_label('Quadrature or Imag')
        self.ax_r.legend()

        # self.ax_r.set_title('Time Domain Waveforms')
        self.ax_r.set_xlabel(self.xylabels[0])
        self.ax_r.set_ylabel(self.xylabels[1])

        self.ax_r.set_ylim(-amp - 2, amp + 2)
        self.ax_r.set_xlim(0, self.max_t)

        self.counter = 0

    def update(self, emitted):

        # self.counter = self.counter + 1

        # drawing the individual signals
        if not pause:

            x_index = emitted[-1]

            # round is crucial here since int will make 199.99 as 199. Want to remove round off errors by rounding
            # to the nearest
            x_index_wrapped = wrap_around(x_index, round(self.max_t / self.dt))

            emitted = emitted[:-1]

            # print('rect values is ', y)
            # round is necessary to insure round off errors don't impact the synced updates between rect plots

            if self.t_data.size == 0:
                last_t = 0
            else:
                last_t = round(self.t_data[-1], 6)

            if continuous:
                if last_t > self.t_data[0] + self.max_t:
                    self.ax_r.set_xlim(last_t - self.max_t, last_t)
            else:
                # frozen frame but keeps drawing on the same frame
                # round is necessary to insure round off errors don't impact the synced updates between rect plots
                round_downed_max_t = round(self.max_t - self.dt, 6)
                if last_t >= round_downed_max_t:
                    self.t_data = np.empty(0)
                    self.y_data_cmbd = [[], []]
                    self.y_data_curr_pt = [0, 0]

            # t = self.t_data[-1] + self.dt
            #
            # # self.t_data.append(t)
            # self.t_data = np.append(self.t_data, t)

            self.t_data = (np.append(self.t_data, x_index_wrapped * self.dt))
            # self.t_data = self.t_data * self.dt

            if x_index_wrapped == 0:
                self.t_data = np.empty(0)
                self.t_data = np.append(self.t_data, x_index_wrapped * self.dt)
                # self.t_data = self.t_data * self.dt

            for emitted_sig, sig_line_r, y_data_1 in zip(emitted, self.sig_lines_r, self.y_data):
                y_data_1[0].append(emitted_sig.real)
                y_data_1[1].append(emitted_sig.imag)

                # these will show the individual line imag and real parts of each signal in rotating_phasors
                # these are usually commented out to avoid clutter
                # sig_line_r[0].set_data(self.t_data, y_data_1[0])
                # sig_line_r[1].set_data(self.t_data, y_data_1[1])

            # combined output drawing
            # if not pause:

            real_list = []
            imag_list = []

            # for emitted_sig in emitted:
            #     real_list.append(emitted_sig.real)
            #     imag_list.append(emitted_sig.imag)

            real_list.append(sum(emitted).real)
            imag_list.append(sum(emitted).imag)

            real_sum = sum(real_list)
            imag_sum = sum(imag_list)

            if x_index_wrapped == 0:
                self.y_data_cmbd[0] = []
                self.y_data_cmbd[1] = []

            self.y_data_cmbd[0].append(real_sum)
            self.y_data_cmbd[1].append(imag_sum)

            y_low, y_high = self.ax_r.get_ylim()

            if len(self.legend_list) == 1:
                max_pt = self.y_data_cmbd[0][-1]
                min_pt = self.y_data_cmbd[0][-1]
            else:
                max_pt = max(self.y_data_cmbd[0][-1], self.y_data_cmbd[1][-1])
                min_pt = min(self.y_data_cmbd[0][-1], self.y_data_cmbd[1][-1])

            if max_pt >= y_high:
                self.ax_r.set_ylim(y_low, max_pt + 10)
            if min_pt <= y_low:
                self.ax_r.set_ylim(min_pt - 10, y_high)

            # if max_pt > y_high or min_pt < y_low:
            #     self.ax_r.set_ylim(-(abs(min_pt) + 10), abs(max_pt) + 10)

            # this will draw the final combined output of the signals in rotating_phasors
            if len(self.legend_list) == 1:
                self.sig_lines_r_cmbd[0].set_data(self.t_data / 1, self.y_data_cmbd[0])
            else:
                self.sig_lines_r_cmbd[0].set_data(self.t_data / 1, self.y_data_cmbd[0])
                self.sig_lines_r_cmbd[1].set_data(self.t_data / 1, self.y_data_cmbd[1])

            # if not pause:
            self.y_data_curr_pt[0] = sum(emitted).real
            self.y_data_curr_pt[1] = sum(emitted).imag

            # this will draw the final combined output of the signals in rotating_phasors
            if len(self.legend_list) == 1:
                self.sig_lines_r_curr_pt[0].set_data(self.t_data[-1], self.y_data_curr_pt[0])
            else:
                self.sig_lines_r_curr_pt[0].set_data(self.t_data[-1], self.y_data_curr_pt[0])
                self.sig_lines_r_curr_pt[1].set_data(self.t_data[-1], self.y_data_curr_pt[1])

        # print(list(flatten(self.sig_lines_r)))
        return list(flatten(self.sig_lines_r + self.sig_lines_r_cmbd + self.sig_lines_r_curr_pt))


class ScopePolarCmbd(object):
    def __init__(self, ax, num_sigs):
        self.ax_p = ax
        self.ax_p.set_rmax(1)
        self.mag_accu = []
        self.theta_accu = []
        self.sig_lines_p = []

        self.mag_accu_cmbd = []
        self.theta_accu_cmbd = []

        # self.lines = [plt.plot([], [])[0] for _ in range(2)]

        # data lines for drawing the real and imag time waves for each item in rotating_phasors
        for _ in range(num_sigs):
            self.sig_lines_p.append(
                [self.ax_p.plot([], [], fstr, linewidth=lw)[0] for _, fstr, lw in
                 zip(range(4), ['-', '-', '-', '-'], [3, 1.5, 1.5, 1.5])])
            self.mag_accu.append([0])
            self.theta_accu.append([0])

        # data lines for drawing the real and imag time waves of combined signals in rotating_phasors
        self.sig_lines_p_cmbd = [self.ax_p.plot([], [], fstr, linewidth=lw)[0] for _, fstr, lw in
                                 zip(range(4), ['g-', '-', 'r-', 'b-'], [3, 1.5, 1.5, 1.5])]

        self.prev_end_pts = 0 + 0j

        # adding legend
        self.sig_lines_p_cmbd[0].set_label('Combined Phasor')
        self.sig_lines_p_cmbd[2].set_label('In-phase or Real Projection')
        self.sig_lines_p_cmbd[3].set_label('Quadrature or Imag Projection')
        self.ax_p.legend(bbox_to_anchor=(2.3, 1), loc="upper right")

        # self.ax_p.set_title('Rotating Phasors in Polar Plot')

    def update(self, emitted):

        # drawing the individual signals
        self.prev_end_pts = 0 + 0j
        emitted_list = []

        if not pause:
            # strip off index, don't need it here
            emitted = emitted[:-1]
            for emitted_sig, sig_line_p, mag_accu, theta_accu in zip(emitted, self.sig_lines_p, self.mag_accu,
                                                                     self.theta_accu):

                emitted_list.append(emitted_sig)

                if spin_orig_center:
                    mag, theta = cm.polar(emitted_sig)
                else:
                    mag, theta = cm.polar(sum(emitted_list))

                # print(theta)
                mag_accu.append(mag)
                theta_accu.append(theta)

                # # adjust polar r limit
                # if mag_accu[-1] > self.ax_p.get_rmax():
                #     self.ax_p.set_rmax(mag_accu[-1] + 1)

                # # clear mag and theta lists if one rotation is complete
                # if theta_accu[-1] > 0 > theta_accu[-2]:
                #     mag_accu.clear()
                #     theta_accu.clear()
                #     mag_accu.append(mag)
                #     theta_accu.append(theta)

                cmplx = cm.rect(mag, theta)
                x = cmplx.real
                y = cmplx.imag

                mag_x, theta_x = cm.polar(complex(x, 0))
                mag_y, theta_y = cm.polar(complex(0, y))

                mag_pep, theta_pep = cm.polar(self.prev_end_pts)
                # rotating phasor
                sig_line_p[0].set_data([theta_pep, theta], [mag_pep, mag])

                # phasor edge tracing
                # sig_line_p[1].set_data(theta_accu, mag_accu)

                # these will draw the real and imag component of each signal in rotating_phasors on the polar plot
                # usually commented out to avoid clutter
                # # projection to real tracing
                # sig_line_p[2].set_data([theta_x, theta_x], [0, mag_x])
                # # projection to imag tracing
                # sig_line_p[3].set_data([theta_y, theta_y], [0, mag_y])

                if not spin_orig_center:
                    self.prev_end_pts = cm.rect(mag, theta)

            # drawing of combined output
            # if not pause:

            mag, theta = cm.polar(sum(emitted))

            self.mag_accu_cmbd.append(mag)
            self.theta_accu_cmbd.append(theta)

            # adjust polar r limit
            if mag > self.ax_p.get_rmax():
                self.ax_p.set_rmax(mag + 1)

            cmplx = cm.rect(mag, theta)
            x = cmplx.real
            y = cmplx.imag
            # print(cmplx)

            mag_x, theta_x = cm.polar(complex(x, 0))
            mag_y, theta_y = cm.polar(complex(0, y))

            # rotating phasor
            self.sig_lines_p_cmbd[0].set_data([theta, theta], [0, mag])

            # phasor edge tracing
            self.sig_lines_p_cmbd[1].set_data(self.theta_accu_cmbd, self.mag_accu_cmbd)

            # projection to real tracing
            self.sig_lines_p_cmbd[2].set_data([theta_x, theta_x], [0, mag_x])
            # projection to imag tracing
            self.sig_lines_p_cmbd[3].set_data([theta_y, theta_y], [0, mag_y])

        return list(flatten(self.sig_lines_p + self.sig_lines_p_cmbd))


class Scope:
    def __init__(self, num_sigs, max_t=1 * T, dt=Ts):
        self.rect_time = ScopeRectCmbd(ax_rect_cmbd, num_sigs, ['Real', 'Imag'], ['Time [s]', 'Amp [V]'], max_t * 1,
                                       dt * 1)
        self.pol1 = ScopePolarCmbd(ax_polar_cmbd, num_sigs)
        self.rect_mag = ScopeRectCmbd(ax_rect_mag, num_sigs, ['Magnitude'], ['Freq', 'Amp [dB]'], max_t * w,
                                      dt * w)
        self.rect_phase = ScopeRectCmbd(ax_rect_phase, num_sigs, ['Phase'], ['Freq', 'Amp [Deg]'], max_t * w,
                                        dt * w)

    def update(self, emitted):

        if not pause:
            x_index = emitted[-1]
        else:
            x_index = 1

        lines_time = self.rect_time.update(emitted)
        polars = self.pol1.update(emitted)

        mag, phase = cm.polar(sum(emitted[:-1]))

        phase = np.rad2deg(phase)

        if mag < 0.01:
            lines_mag = self.rect_mag.update([20 * np.log10(0.01), x_index])
            lines_phase = self.rect_phase.update([phase, x_index])
        else:
            lines_mag = self.rect_mag.update([20 * np.log10(mag), x_index])
            lines_phase = self.rect_phase.update([phase, x_index])

        # lines_mag = self.rect_mag.update(emitted)
        # lines_phase = self.rect_phase.update(emitted)

        # return list(flatten(polars + lines_mag + lines_phase))

        return list(flatten(lines_time + polars + lines_mag + lines_phase))

        # return list(flatten(lines_time + polars))


def sig_emitter():
    i = 0
    curr_phasor_values = []
    while i < x_t.size:
        # print('i is ', i)
        if not pause:
            curr_phasor_values.clear()
            for phasor in rotating_phasors:
                curr_phasor_values.append(phasor[i])

            curr_phasor_values.append(i)
            i = i + 1

        yield curr_phasor_values


def onClick(event):
    global pause
    pause ^= True


fig = plt.figure(1, figsize=(10, 10))

ax_polar_cmbd = plt.subplot(3, 2, 1, projection='polar')

ax_rect_cmbd = plt.subplot(3, 2, 2)

# fft plot of the sum of signals in rotating_phasors
sum_sig = sum(rotating_phasors)
yf = fft(sum_sig)
xf = fftfreq(num_sampls, 1 / fs)
ax_rect_fft = plt.subplot(3, 2, 3)
ax_rect_fft.set_xlabel('Frequency [kHz]')
ax_rect_fft.set_ylabel('Magnitude')
xf = fftshift(xf)
yf = fftshift(yf)
ax_rect_fft.plot(xf / 1e3, 1.0 / num_sampls * np.abs(yf))

# ax_mag = plt.subplot(3, 2, 4)
ax_rect_mag = plt.subplot(3, 2, 4)
ax_rect_phase = plt.subplot(3, 2, 5)

scope_main = Scope(len(rotating_phasors))

interval = 10
fig.canvas.mpl_connect('key_press_event', onClick)

# pass a generator in "sineEmitter" to produce data for the update func
ani = animation.FuncAnimation(fig, scope_main.update, sig_emitter, interval=interval,
                              blit=True)

plt.show()
