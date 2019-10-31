"""
@author: Kaung Myat San Oo
Animation for understanding rotating phasors, negative frequency in DSP concepts

Add, remove, or modify signals in cmplx_exps
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
from scipy.fftpack import fft, fftshift, fftfreq

pause = False

# Your Parameters
amp = 1  # 1V        (Amplitude)
f = 1000  # 1kHz      (Frequency)
fs = 200000  # 200kHz    (Sample Rate)
T = 1 / f
Ts = 1 / fs
N = fs  # number of samples

x_t = np.arange(0, fs * Ts, Ts)
pi = np.pi

# Select if you want to display the sine as a continuous wave
#  True = continuous (not able to zoom in x-direction)
#  False = Non-continuous  (able to zoom)
continuous = True

# if set True, all phasors in cmplx_exps will spin with respect to center of polar plot
# if False, all phasors will spin with respect to the end of the previous phasor end point (true vector addition)
spin_orig_center = False

x = np.arange(N)

phi = 0

cmplx_exps = [
    np.ones(N),
    # np.array([amp * np.exp(1j * (2 * pi * f * (i / N) + phi)) for i in x]),
    np.array([amp * 1 * np.exp(1j * (2 * pi * (-f / 1.1) * (i / N) + phi)) for i in x])
]

# cmplx_exps = [
#     # np.ones(N),
#     # np.array([amp * np.exp(1j * (2 * pi * f * (i / N) + phi)) for i in x]),
#     np.array([amp * 1 * np.exp(1j * (2 * pi * (-f / 1) * (i / N) + phi)) for i in x]),
#     np.array([amp * 1 * np.exp(1j * (2 * pi * (-f / 2) * (i / N) + phi)) for i in x]),
#     np.array([amp * 1 * np.exp(1j * (2 * pi * (-f / 3) * (i / N) + phi)) for i in x]),
#     np.array([amp * 1 * np.exp(1j * (2 * pi * (-f / 4) * (i / N) + phi)) for i in x]),
#     np.array([amp * 1 * np.exp(1j * (2 * pi * (-f / 5) * (i / N) + phi)) for i in x]),
#     np.array([amp * 1 * np.exp(1j * (2 * pi * (-f / 6) * (i / N) + phi)) for i in x]),
#
# ]


# cmplx_exps = [np.array([amp * np.exp(1j * (2 * pi * f * (i / fs) + phi)) for i in x])

# flatten list items
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


class ScopeRectCmbd(object):
    def __init__(self, num_sigs, max_t=2 * T, dt=Ts):
        self.ax_r = ax_rect_cmbd
        self.ax_r.grid(True)
        self.dt = dt
        self.max_t = max_t
        self.t_data = np.zeros(1)
        self.y_data = []
        self.sig_lines_r = []

        # data lines for drawing the phasor, edge tracing, real and image projection for each item in cmplx_exps
        for _ in range(num_sigs):
            self.sig_lines_r.append([self.ax_r.plot([], [], fstr)[0] for _, fstr in zip(range(2), ['', ''])])
            self.y_data.append([[0], [0]])

        # data lines for drawing the phasor, edge tracing, real and image projection for combined output signals in
        # cmplx_exps
        self.sig_lines_r_cmbd = [self.ax_r.plot([], [], fstr, linewidth=lw,
                                                path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()])[0]
                                 for _, fstr, lw in
                                 zip(range(2), ['r-', 'b-'], [3, 2])]
        self.y_data_cmbd = [[0], [0]]

        # adding legend
        self.sig_lines_r_cmbd[0].set_label('In-phase or Real')
        self.sig_lines_r_cmbd[1].set_label('Quadrature or Imag')
        self.ax_r.legend()

        # self.ax_r.set_title('Time Domain Waveforms')
        self.ax_r.set_ylabel('Amplitude [V]')
        self.ax_r.set_xlabel('Time [s]')

        self.ax_r.set_ylim(-amp - 2, amp + 2)
        self.ax_r.set_xlim(0, self.max_t)

    def update(self, emitted):

        # drawing the individual signals
        if not pause:
            # print('rect values is ', y)
            last_t = self.t_data[-1]
            if continuous:
                if last_t > self.t_data[0] + self.max_t:
                    self.ax_r.set_xlim(last_t - self.max_t, last_t)

            t = self.t_data[-1] + self.dt

            # self.t_data.append(t)
            self.t_data = np.append(self.t_data, t)

            for emitted_sig, sig_line_r, y_data_1 in zip(emitted, self.sig_lines_r, self.y_data):
                y_data_1[0].append(emitted_sig.real)
                y_data_1[1].append(emitted_sig.imag)

                # these will show the individual line imag and real parts of each signal in cmplx_exps
                # these are usually commented out to avoid clutter
                # sig_line_r[0].set_data(self.t_data, y_data_1[0])
                # sig_line_r[1].set_data(self.t_data, y_data_1[1])

        # combined output drawing
        if not pause:

            real_list = []
            imag_list = []

            for emitted_sig in emitted:
                real_list.append(emitted_sig.real)
                imag_list.append(emitted_sig.imag)

            real_sum = sum(real_list)
            imag_sum = sum(imag_list)

            self.y_data_cmbd[0].append(real_sum)
            self.y_data_cmbd[1].append(imag_sum)

            # this will draw the final combined output of the signals in cmplx_exps
            self.sig_lines_r_cmbd[0].set_data(self.t_data / 1, self.y_data_cmbd[0])
            self.sig_lines_r_cmbd[1].set_data(self.t_data / 1, self.y_data_cmbd[1])

        # print(list(flatten(self.sig_lines_r)))
        return list(flatten(self.sig_lines_r + self.sig_lines_r_cmbd))


class ScopePolarCmbd(object):
    def __init__(self, num_sigs):
        self.ax_p = ax_polar_cmbd
        self.ax_p.set_rmax(1)
        self.mag_accu = []
        self.theta_accu = []
        self.sig_lines_p = []

        # self.lines = [plt.plot([], [])[0] for _ in range(2)]

        # data lines for drawing the real and imag time waves for each item in cmplx_exps
        for _ in range(num_sigs):
            self.sig_lines_p.append(
                [self.ax_p.plot([], [], fstr, linewidth=lw)[0] for _, fstr, lw in
                 zip(range(4), ['-', '-', '-', '-'], [3, 1.5, 1.5, 1.5])])
            self.mag_accu.append([0])
            self.theta_accu.append([0])

        # data lines for drawing the real and imag time waves of combined signals in cmplx_exps
        self.sig_lines_p_cmbd = [self.ax_p.plot([], [], fstr, linewidth=lw)[0] for _, fstr, lw in
                                 zip(range(4), ['g-', '.', 'r-', 'b-'], [3, 1.5, 1.5, 1.5])]

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
        for emitted_sig, sig_line_p, mag_accu, theta_accu in zip(emitted, self.sig_lines_p, self.mag_accu,
                                                                 self.theta_accu):
            if not pause:
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

                # clear mag and theta lists if one rotation is complete
                if theta_accu[-1] > 0 > theta_accu[-2]:
                    mag_accu.clear()
                    theta_accu.clear()
                    mag_accu.append(mag)
                    theta_accu.append(theta)

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

                # these will draw the real and imag component of each signal in cmplx_exps on the polar plot
                # usually commented out to avoid clutter
                # # projection to real tracing
                # sig_line_p[2].set_data([theta_x, theta_x], [0, mag_x])
                # # projection to imag tracing
                # sig_line_p[3].set_data([theta_y, theta_y], [0, mag_y])

                if not spin_orig_center:
                    self.prev_end_pts = cm.rect(mag, theta)

        # drawing of combined output
        if not pause:

            mag, theta = cm.polar(sum(emitted))

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
            # self.sig_lines_p_cmbd[1].set_data(theta_accu, mag_accu)

            # projection to real tracing
            self.sig_lines_p_cmbd[2].set_data([theta_x, theta_x], [0, mag_x])
            # projection to imag tracing
            self.sig_lines_p_cmbd[3].set_data([theta_y, theta_y], [0, mag_y])

        return list(flatten(self.sig_lines_p + self.sig_lines_p_cmbd))


class Scope(ScopeRectCmbd, ScopePolarCmbd):
    def __init__(self, num_sigs, max_t=4 * T, dt=Ts):
        ScopeRectCmbd.__init__(self, num_sigs, max_t, dt)
        ScopePolarCmbd.__init__(self, num_sigs)

    def update(self, emitted):
        lines = ScopeRectCmbd.update(self, emitted)
        polars = ScopePolarCmbd.update(self, emitted)

        return list(flatten(lines + polars))


def sine_emitter():
    i = 0
    real = 0
    imag = 0
    cmplx_exp_real_imag = []
    while i < x.size:
        # print('i is ', i)
        if not pause:
            cmplx_exp_real_imag.clear()
            for cmplx_exp in cmplx_exps:
                cmplx_exp_real_imag.append(cmplx_exp[i])

            i = i + 1

        yield cmplx_exp_real_imag


def onClick(event):
    global pause
    pause ^= True


fig = plt.figure(1, figsize=(10, 10))

ax_polar_cmbd = plt.subplot(3, 1, 1, projection='polar')

ax_rect_cmbd = plt.subplot(3, 1, 2)

# fft plot of the sum of signals in cmplx_exps
sum_sig = sum(cmplx_exps)
yf = fft(sum_sig)
xf = fftfreq(N, 1 / fs)
ax_rect_fft = plt.subplot(3, 1, 3)
ax_rect_fft.set_xlabel('Frequency [kHz]')
ax_rect_fft.set_ylabel('Magnitude')
xf = fftshift(xf)
yf = fftshift(yf)
ax_rect_fft.plot(xf / 1e3, 1.0 / N * np.abs(yf))

scope_main = Scope(len(cmplx_exps))

interval = 10

fig.canvas.mpl_connect('key_press_event', onClick)

# pass a generator in "sineEmitter" to produce data for the update func
ani = animation.FuncAnimation(fig, scope_main.update, sine_emitter, interval=interval,
                              blit=True)

plt.show()
