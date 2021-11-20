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
import custom_tools.fftplot as fftplot
import scipy.signal as sig

pi = np.pi


def fftfreq1(N, d):
    # this function is different from fftfreq from scipy
    # in that it returns the first N/2+1 plus the second half (N/2-1 in scipy edition) as fftfreq for even N
    # for odd N, it returns the first (N+1)/2 plus the second half ((N-1/)2 in scipy edition)
    # (this is how Richard Lyon's DSP book states)

    if N % 2 == 0:
        # even
        a1 = np.arange(0, N / 2 + 1, 1)
        a2 = np.arange(-N / 2 + 1, 0, 1)
        return np.concatenate((a1, a2)) / (N * d)
    else:
        # odd
        a1 = np.arange(0, (N + 1) / 2, 1)
        a2 = np.arange(-(N - 1) / 2, 0, 1)
        return np.concatenate((a1, a2)) / (N * d)


def angle2(x):
    fin_res = []

    for i in x:
        imag = i.imag
        real = i.real

        if real == 0 and isinstance(real, float):
            real = 0

        if imag == 0 and isinstance(real, float):
            imag = 0

        res = np.arctan2(imag, real)

        fin_res.append(res)

    return np.array(fin_res)


def wrap_around(x, count):
    return x % count


def flatten(lis):
    # flatten list items
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def coherency_calc(fin, fs, N):
    M = np.floor(fin / fs * N)
    if M % 2 == 0:
        print('M is even')
        M = M + 1  # M must be odd

    if M < 1:
        print('Cant calculate f_coherent')
        f_coherent = fin
    else:
        f_coherent = M / N * fs

    return f_coherent, M


def returnSciNotation(x):
    x = format(x, 'e')
    num, pow10 = x.split('e')

    num = float(num)

    pow10 = float(pow10)

    return num, pow10


# ********************* choose operating mode here *********************
# direct_phasor:    Enter time-domain signals (such as real cosine, complex cosine) in rotating_phasors.
#                   See predefined examples below.
# FT_mode:          Enter time domain samples (real or complex) in input_vector and the polar plot will
#                   show reconstructed time domain samples
# FIR_mode:         Enter FIR filter coefficients h[k] or bk in input vector. polar plot will show how frequency
#                   response of FIR filter is made
available_op_modes = ['direct_phasor', 'FT_mode', 'FIR_mode']
op_mode = 'FT_mode'

Fs = 80e3  # 200kHz    (Sample Rate)

Ts = 1 / Fs

Ts_sci, Ts_pow10 = returnSciNotation(Ts)

# num_sampls = Fs  # number of samples
num_sampls = 2 ** 12  # number of samples

# f1, _ = coherency_calc(f1, Fs, num_sampls)

x_t = np.arange(0, num_sampls * Ts, Ts)
n = x_t

# list of phasor arrays that get passed into animation function
rotating_phasors = []

pause = False

# Select if you want to display the sine as a continuous wave
#  True = continuous (not able to zoom in x-direction)
#  False = Non-continuous  (able to zoom)
continuous = False

# if set True, all phasors in rotating_phasors will spin with respect to center of polar plot
# if False, all phasors will spin with respect to the end of the previous phasor end point (true vector addition)
spin_orig_center = False

double_sided_FFT = True

# turning coherency on will drift the rect displays, but produces clean FFTs
# This happens when Fs/fund is not an integer since the coherency works only for whole window length (num_sampls),
# and rect plot shows only one cycle
coherency = True

# this data is from noise cancellation filter
# input_vector = np.array([0.13083826, 0.12517881, 0.11899678, 0.11231907, 0.10508106,
#                          0.09738001, 0.08925131, 0.08073239, 0.07186244, 0.06268219,
#                          0.05323363, 0.04355975, 0.03370433, 0.02371163, 0.01362624,
#                          0.00349278, -0.00664429, -0.01674086, -0.0267534, -0.03663912])

input_vector = np.array([1, -1, -1 + 2j])
N = len(input_vector)

max_mag = np.absolute(input_vector).max()
max_theta = np.angle(input_vector, deg=True).min()

amp1 = 1  # 1V        (Amplitude)
f1 = 1e3  # 1kHz      (Frequency)
phi1 = 0

amp2 = 1
f2 = 2e3
phi2 = 0

if coherency:
    f1, _ = coherency_calc(f1, Fs, num_sampls)
    f2, _ = coherency_calc(f2, Fs, num_sampls)
    f3, _ = coherency_calc(20e6, Fs, num_sampls)

fund_period = 1 / f1
w0 = 2 * pi * f1

# for complex down conversion example, shift amount, LO_freq, needed for center of band to be centered at DC
LO_freq, _ = coherency_calc(f2 - f1 / 2, Fs, num_sampls)

if op_mode == 'direct_phasor':
    # populate rotating_phasors with desired sinusoids
    rotating_phasors = [
        # Rick Lyon DFT Example, fs = 8kHz, f1 = 1kHz
        # np.sin(2 * pi * f1 * x_t),
        # 0.5 * np.sin(2 * pi * f1 * 2 * x_t + 3 * pi / 4),

        # complex cosine
        # amp1 / 1 * np.exp(1j * (2 * pi * f1 * x_t + phi1)),

        # complex cosine with noise
        # amp1 / 1 * np.exp(1j * (2 * pi * f1 * x_t + phi1)),
        # 0.1 * 1j * np.random.randn(num_sampls),
        # 0.1 * 1 * np.random.randn(num_sampls),

        # 9V battery: making 9V DC with phase.
        # 9 * np.exp(1j * (2 * pi * 0 * x_t + pi)),

        # real cosine
        amp1 / 2 * np.exp(1j * (2 * pi * f1 * x_t + phi1)),
        amp1 / 2 * np.exp(-1j * (2 * pi * f1 * x_t + phi1)),

        # real cosine, no phasors
        # amp1 * np.cos(2 * pi * f1 * x_t + phi1),

        # real sine 1
        # amp1 / 2j * np.exp(1j * (2 * pi * f1 * x_t + phi1)),
        # -amp1 / 2j * np.exp(-1j * (2 * pi * f1 * x_t + phi1)),

        # real sine 2
        # amp2 / 2j * np.exp(1j * (2 * pi * f2 * x_t + phi2)),
        # -amp2 / 2j * np.exp(-1j * (2 * pi * f2 * x_t + phi2))

        # real sine in sine form, not euler's form
        # amp2 * np.sin(2 * pi * f2 * x_t + phi2)

        # # I-jQ (complex down conversion eg)
        # # f1 and f2 signify the lower and upper freq of a bandpass signal
        # # phase plot should be clean if all frequencies are coherent
        # (amp2 * np.cos(2 * pi * f2 * x_t + phi2) + 0.5 * np.cos(2 * pi * f1 * x_t + phi2))
        # * np.cos(2 * pi * LO_freq * x_t + phi2),
        # -1j * (amp2 * np.cos(2 * pi * f2 * x_t + phi2) + 0.5 * np.cos(2 * pi * f1 * x_t + phi2))
        # * np.sin(2 * pi * LO_freq * x_t + phi2)

    ]

elif op_mode == 'FT_mode':
    # if FT_mode:
    # in this mode, input vector is time domain samples (real or complex) and
    # the polar plot will show reconstructed time domain samples
    xn = np.array(input_vector)
    Xk = fft(xn)

    if double_sided_FFT:
        freq_idx = fftfreq1(N, 1 / N)
    else:
        freq_idx = np.arange(0, N, 1)
    # Xk = fftshift(Xk)
    # freq_idx = fftshift(freq_idx)

    # inverse FFT to check, should be the same as input_vector
    inverseFFT = ifft(Xk)
    # rewriting inverse DFT equation
    # xn = 1/N summation k=0 to N-1 {Xk * e^(j * 2pi * k * n / N)
    for k, X_curr_k in zip(freq_idx, Xk):
        rotating_phasors.append(1 / N * X_curr_k *
                                np.array(np.exp(
                                    1j * (2 * pi * f1 * k * n / 1))))  # 1/N term is not included, it's just speed

elif op_mode == 'FIR_mode':
    # in this mode, input vector is FIR filter coefficients h[k] or bk
    # polar plot will show how frequency response of FIR filter is made
    filt_coeffs = np.array(input_vector)

    # rewriting frequency response of FIR filter
    # H(e^jw) = summation k=0 to M-1 {bk e^(-j * w0 * k)}
    for k, b_curr_k in enumerate(filt_coeffs):
        rotating_phasors.append(b_curr_k *
                                np.array(np.exp(-1j * (w0 * k * n))))
else:
    raise NotImplementedError(f'Please choose one of {available_op_modes}.')

num_sigs = len(rotating_phasors)


class ScopeRectCmbd(object):
    def __init__(self, ax, num_sigs, legend_list, xylabels, max_t=2 * fund_period, dt=Ts):
        self.ax_r = ax
        self.ax_r.grid(True)
        self.dt = dt
        self.max_t = max_t
        self.t_data = np.empty(0)
        self.y_data = []
        self.sig_lines_r = []
        self.legend_list = legend_list
        self.xylabels = xylabels
        self.num_sigs = num_sigs

        # data lines for drawing the real and imag time waves for each item in rotating_phasors
        # for each signal in rotating_phasors, sig_lines_r will draw the real and imag projections in rect plot
        for _ in range(num_sigs):
            self.sig_lines_r.append([self.ax_r.plot([], [], fstr)[0] for _, fstr in zip(range(2), ['', ''])])
            self.y_data.append([[0], [0]])

        # data lines for drawing the real and imag time waves for each item in rotating_phasors
        # sig_lines_r_cmbd will draw the real and imag of all signals combined in rotating_phasors
        self.sig_lines_r_cmbd = [self.ax_r.plot([], [], fstr, linewidth=lw,
                                                path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()])[0]
                                 for _, fstr, lw in
                                 zip(range(2), ['r-', 'b-'], [3, 2])]

        self.y_data_cmbd = [[], []]

        # sig_lines_r_curr_pt will draw the current location of the projection with an x
        self.sig_lines_r_curr_pt = [self.ax_r.plot([], [], fstr, linewidth=lw,
                                                   path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()])[
                                        0]
                                    for _, fstr, lw in
                                    zip(range(2), ['rx', 'bx'], [3, 2])]
        self.y_data_curr_pt = [0, 0]

        # # adding legend
        for line, legend in zip(self.sig_lines_r_cmbd, self.legend_list):
            line.set_label(legend)
        self.ax_r.legend()

        self.ax_r.set_xlabel(self.xylabels[0])
        self.ax_r.set_ylabel(self.xylabels[1])

        self.ax_r.set_ylim(-amp1 - 2, amp1 + 2)
        self.ax_r.set_xlim(0, self.max_t)

    def update(self, emitted):

        round_dec = 6
        # drawing the individual signals
        if not pause:

            x_index = emitted[-1]

            print('****** emitted ', emitted)

            # round is crucial here since int will make 199.99 as 199. Want to remove round off errors by rounding
            # to the nearest
            x_index_wrapped = wrap_around(x_index, round(self.max_t / self.dt))
            print('x_index ', x_index)
            print('x wrapped ', x_index_wrapped)

            # strip off index since it's been saved
            emitted = emitted[:-1]

            # round is necessary to insure round off errors don't impact the synced updates between rect plots
            if self.t_data.size == 0:
                last_t = 0
            else:
                last_t = round(self.t_data[-1], round_dec)

            if continuous:
                if self.t_data.size == 0:
                    if last_t > self.max_t:
                        self.ax_r.set_xlim(last_t - self.max_t, last_t)
                else:
                    if last_t > self.t_data[0] + self.max_t:
                        self.ax_r.set_xlim(last_t - self.max_t, last_t)
            else:
                # frozen frame but keeps drawing on the same frame
                # round is necessary to insure round off errors don't impact the synced updates between rect plots
                round_nearest_max_t = round(self.max_t - self.dt, round_dec)
                print('self dt is', self.dt)
                print('last_t is ', last_t)
                print('max_t is ', self.max_t)
                print('round_nearest_max_t is ', round_nearest_max_t)

                if last_t >= round_nearest_max_t:
                    # print('last_t is ', last_t)
                    # print('round_nearest_max_t is ', round_nearest_max_t)

                    self.t_data = np.empty(0)
                    self.y_data_cmbd = [[], []]
                    self.y_data_curr_pt = [0, 0]

                    print('t_data ', self.t_data)
                    print('y_data_cmbd_0', self.y_data_cmbd[0])

            if x_index_wrapped == 0:
                # reset t_data once one cycle (2pi rotation) is completed
                self.t_data = np.empty(0)

                # reinitialize y_data_cmbd to start clean slate if x_index is zero
                self.y_data_cmbd[0] = []
                self.y_data_cmbd[1] = []

                # reinitialize y_data_cmbd to start clean slate if x_index is zero
                self.y_data = []
                for _ in range(self.num_sigs):
                    self.y_data.append([[], []])

            self.t_data = (np.append(self.t_data, x_index_wrapped * self.dt))

            for emitted_sig, sig_line_r, y_data_1 in zip(emitted, self.sig_lines_r, self.y_data):
                y_data_1[0].append(emitted_sig.real)
                y_data_1[1].append(emitted_sig.imag)

                # these will show the individual line imag and real parts of each signal in rotating_phasors
                # these are usually commented out to avoid clutter
                # sig_line_r[0].set_data(self.t_data, y_data_1[0])
                # sig_line_r[1].set_data(self.t_data, y_data_1[1])

            # combined output drawing
            curr_pt_real = sum(emitted).real
            curr_pt_imag = sum(emitted).imag

            self.y_data_cmbd[0].append(curr_pt_real)
            self.y_data_cmbd[1].append(curr_pt_imag)

            self.y_data_curr_pt[0] = curr_pt_real
            self.y_data_curr_pt[1] = curr_pt_imag

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

            # this will draw the final combined output of the signals in rotating_phasors
            if len(self.legend_list) == 1:
                # only 1 line will be drawn, it'll be either mag or phase
                self.sig_lines_r_cmbd[0].set_data(self.t_data / 1, self.y_data_cmbd[0])
            else:
                # two lines will be drawn, it's real projection and imag projection
                self.sig_lines_r_cmbd[0].set_data(self.t_data / 1, self.y_data_cmbd[0])
                self.sig_lines_r_cmbd[1].set_data(self.t_data / 1, self.y_data_cmbd[1])

                # print('t_data')
                # print(self.t_data)
                # print('data0')
                # print(self.y_data_cmbd[0])
                # print('data1')
                # print(self.y_data_cmbd[1])

            # this will draw the current point of final combined output of the signals in rotating_phasors
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
        if self.ax_p.name == 'polar':
            self.polar_plot = True
        else:
            self.polar_plot = False
            self.ax_p.grid(True)
            self.ax_p.set_aspect('equal', adjustable='box')

        self.mag_accu = []
        self.theta_accu = []
        self.sig_lines_p = []

        self.mag_accu_cmbd = []
        self.theta_accu_cmbd = []

        # data lines for drawing the phasor, edge tracing, real and image projection for each item in rotating_phasors
        for _ in range(num_sigs):
            self.sig_lines_p.append(
                [self.ax_p.plot([], [], fstr, linewidth=lw)[0] for _, fstr, lw in
                 zip(range(4), ['-', '-', '-', '-'], [3, 1.5, 1.5, 1.5])])
            self.mag_accu.append([0])
            self.theta_accu.append([0])

        # data lines for drawing the combined phasor, edge tracing, real and image projection
        # or each item in rotating_phasors
        self.sig_lines_p_cmbd = [self.ax_p.plot([], [], fstr, linewidth=lw)[0] for _, fstr, lw in
                                 zip(range(4), ['g-', '-', 'r-', 'b-'], [3, 1.5, 1.5, 1.5])]

        self.prev_end_pts = 0 + 0j

        # adding legend
        self.sig_lines_p_cmbd[0].set_label('Combined Phasor')
        self.sig_lines_p_cmbd[2].set_label('In-phase or Real Projection')
        self.sig_lines_p_cmbd[3].set_label('Quadrature or Imag Projection')

        if self.polar_plot:
            self.ax_p.legend(bbox_to_anchor=(2.3, 1), loc="upper right")
        else:
            # self.ax_p.legend()
            self.ax_p.legend(bbox_to_anchor=(2.3, 1), loc="upper right")

        # self.ax_p.set_title('Rotating Phasors in Polar Plot')

        if self.polar_plot:
            self.ax_p.set_rmax(max_mag + 2)
        else:
            min_xy = round(-max_mag - 1)
            max_xy = round(max_mag + 1)
            self.ax_p.set_xlim(min_xy, max_xy)
            self.ax_p.set_ylim(min_xy, max_xy)
            #
            # self.ax_p.set_xticks(np.arange(min_xy, max_xy, step=round((max_xy-min_xy)/5)))
            # self.ax_p.set_yticks(np.arange(min_xy, max_xy, step=round((max_xy-min_xy)/5)))
            pass

        self.one_cycle_index = Fs / f1

    def update(self, emitted):

        # drawing the individual signals
        self.prev_end_pts = 0 + 0j
        emitted_list = []

        if not pause:
            x_index = emitted[-1]
            x_index_wrapped = wrap_around(x_index, self.one_cycle_index)

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
                if x_index_wrapped == 0:
                    mag_accu = []
                    theta_accu = []

                mag_accu.append(mag)
                theta_accu.append(theta)

                cmplx = cm.rect(mag, theta)
                x = cmplx.real
                y = cmplx.imag

                mag_x, theta_x = cm.polar(complex(x, 0))
                mag_y, theta_y = cm.polar(complex(0, y))

                mag_pep, theta_pep = cm.polar(self.prev_end_pts)

                # rotating phasor
                if self.polar_plot:
                    sig_line_p[0].set_data([theta_pep, theta], [mag_pep, mag])
                else:
                    rect_pep = cm.rect(mag_pep, theta_pep)
                    rect = cm.rect(mag, theta)

                    sig_line_p[0].set_data([rect_pep.real, rect.real], [rect_pep.imag, rect.imag])

                # phasor edge tracing
                # usually commented out to avoid clutter
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
            mag, theta = cm.polar(sum(emitted))

            if x_index_wrapped == 0:
                self.mag_accu_cmbd = []
                self.theta_accu_cmbd = []

            self.mag_accu_cmbd.append(mag)
            self.theta_accu_cmbd.append(theta)

            # adjust polar r limit
            if self.polar_plot:
                if mag >= self.ax_p.get_rmax():
                    self.ax_p.set_rmax(mag + 1)
            else:
                if mag >= self.ax_p.get_xlim()[1] or mag >= self.ax_p.get_ylim()[1]:
                    self.ax_p.set_xlim(round(-mag - 2), round(mag + 2))
                    self.ax_p.set_ylim(round(-mag - 2), round(mag + 2))
            cmplx = cm.rect(mag, theta)
            x = cmplx.real
            y = cmplx.imag
            # print(cmplx)

            mag_x, theta_x = cm.polar(complex(x, 0))
            mag_y, theta_y = cm.polar(complex(0, y))

            if self.polar_plot:
                # rotating phasor
                self.sig_lines_p_cmbd[0].set_data([theta, theta], [0, mag])

                # # phasor edge tracing
                self.sig_lines_p_cmbd[1].set_data(self.theta_accu_cmbd, self.mag_accu_cmbd)
                #
                # # # projection to real tracing
                self.sig_lines_p_cmbd[2].set_data([theta_x, theta_x], [0, mag_x])
                # # # projection to imag tracing
                self.sig_lines_p_cmbd[3].set_data([theta_y, theta_y], [0, mag_y])

            else:
                # rect_pep = cm.rect(mag, theta)
                rect = cm.rect(mag, theta)

                self.sig_lines_p_cmbd[0].set_data([0, rect.real], [0, rect.imag])

                theta_accu_cmbd = np.array(self.theta_accu_cmbd)
                mag_accu_cmbd = np.array(self.mag_accu_cmbd)
                x1 = mag_accu_cmbd * np.cos(theta_accu_cmbd)
                y1 = mag_accu_cmbd * np.sin(theta_accu_cmbd)
                self.sig_lines_p_cmbd[1].set_data(x1, y1)

                self.sig_lines_p_cmbd[2].set_data([0, x], [0, 0])

                self.sig_lines_p_cmbd[3].set_data([0, 0], [0, y])

            # plt.draw()
        return list(flatten(self.sig_lines_p + self.sig_lines_p_cmbd))


class Scope:
    def __init__(self, num_sigs, max_t=1 * fund_period, dt=Ts):
        self.rect_time = ScopeRectCmbd(ax_rect_cmbd, num_sigs, ['Real', 'Imag'],
                                       ['Time [xE{} sec]'.format(round(Ts_pow10)), 'Amp [V]'],
                                       max_t / 10 ** Ts_pow10,
                                       dt / 10 ** Ts_pow10)
        self.pol1 = ScopePolarCmbd(ax_polar_cmbd, num_sigs)
        self.rect_mag = ScopeRectCmbd(ax_rect_mag, num_sigs, ['Magnitude'], ['Freq', 'Amp [dB]'], max_t * w0,
                                      dt * w0)
        self.rect_phase = ScopeRectCmbd(ax_rect_phase, num_sigs, ['Phase'], ['Freq', 'Amp [Deg]'], max_t * w0,
                                        dt * w0)

    def update(self, emitted):

        if not pause:
            x_index = emitted[-1]
        else:
            x_index = 1

        lines_time = self.rect_time.update(emitted)
        polars = self.pol1.update(emitted)

        mag, phase = cm.polar(sum(emitted[:-1]))

        phase = np.rad2deg(phase)

        if mag < 0.001:
            lines_mag = self.rect_mag.update([20 * np.log10(0.001), x_index])
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
# ax_polar_cmbd = plt.subplot(3, 2, 1)
# ax_polar_cmbd.set_xlim(-10, 10)
# ax_polar_cmbd.set_ylim(-10, 10)


if op_mode == 'FT_mode':
    # plot the input_vector points on polar plot
    for point in input_vector:
        if ax_polar_cmbd.name == 'polar':
            mag, theta = cm.polar(point)
            ax_polar_cmbd.plot(theta, mag, 'b*')
        else:
            ax_polar_cmbd.plot(point.real, point.imag, 'b*')

ax_rect_cmbd = plt.subplot(3, 2, 2)

# fft plot of the sum of signals in rotating_phasors
sum_sig = sum(rotating_phasors)

# some random signal
sum_sig1 = np.sin(2 * pi * f1 * x_t) + 0.5 * np.sin(2 * pi * 2000 * x_t + 3 * pi / 4)

# only window data, not zero padded signal:
# win_len = input_len if n_samp > input_len else n_samp

win_len = num_sampls
beta = 12
win = sig.kaiser(win_len, beta)
winscale = np.sum(win)

win_data = win * sum_sig

yf = fft(sum_sig)
xf = fftfreq1(num_sampls, 1 / Fs)
ax_rect_fft = plt.subplot(3, 2, 3)
ax_rect_fft.set_xlabel('Frequency [kHz]')
ax_rect_fft.set_ylabel('Magnitude')
xf = fftshift(xf)
yf = fftshift(yf)

# ax_rect_fft.stem(xf / 1e3, 1.0 / 1 * np.abs(yf), use_line_collection=True)

# extract the first 10% of the sig len
sumg_sig_10pct = sum_sig[0:round(len(sum_sig) * 0.1) + 1]

# fft magnitude scaling depending whether input is real or complex
if np.all(np.isreal(np.round(sumg_sig_10pct, 3))):
    # if input is real
    ax_rect_fft.stem(xf / 1e3, 2 / num_sampls * np.abs(yf), use_line_collection=True)
else:
    # if input is complex
    # do not need 2/N since there's already a 1/2 factor
    # in sinx = (e^jx - e^-jx) / 2j and cox = (e^jx + e^-jx) / 2
    # for complex sig, it's just e^jx, not e^jx / 2
    # Ensure there's no /2 in rotating_phasors if giving complex input
    ax_rect_fft.stem(xf / 1e3, 1.0 / num_sampls * np.abs(yf), use_line_collection=True)

ax_rect_fft = plt.subplot(3, 2, 5)
ax_rect_fft.stem(xf / 1e3, angle2(np.round(yf, 1)) * 180 / pi, use_line_collection=True)
ax_rect_fft.set_xlabel('Frequency [kHz]')
ax_rect_fft.set_ylabel('Phase [Deg]')

# ax_mag = plt.subplot(3, 2, 4)
ax_rect_mag = plt.subplot(3, 2, 4)
ax_rect_phase = plt.subplot(3, 2, 6)

scope_main = Scope(len(rotating_phasors))

interval = 4000 / (Fs / f1)
fig.canvas.mpl_connect('key_press_event', onClick)

# pass a generator in "sineEmitter" to produce data for the update func
ani = animation.FuncAnimation(fig, scope_main.update, sig_emitter, interval=interval,
                              blit=True)

plt.show()
