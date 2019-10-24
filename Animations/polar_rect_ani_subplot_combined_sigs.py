"""
@author: Kaung Myat San Oo
Script for understanding rotating phasors, negative frequency in DS concepts

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import cmath as cm
from collections.abc import Iterable
import matplotlib.patheffects as pe

pause = False

# Your Parameters
amp = 1  # 1V        (Amplitude)
f = 1000  # 1kHz      (Frequency)
fs = 200000  # 200kHz    (Sample Rate)
T = 1 / f
Ts = 1 / fs

x_t = np.arange(0, fs * Ts, Ts)
pi = np.pi

# Select if you want to display the sine as a continuous wave
#  True = continuous (not able to zoom in x-direction)
#  False = Non-continuous  (able to zoom)
continuous = True

x = np.arange(fs)

phi = pi / 4

cmplx_exps = [np.array([amp * np.exp(1j * (2 * pi * f * (i / fs) + phi)) for i in x]),
              np.array([amp * 1 * np.exp(1j * (2 * pi * (-f / 1) * (i / fs) + phi)) for i in x])
              ]


# cmplx_exps = [np.array([amp * np.exp(1j * (2 * pi * f * (i / fs) + phi)) for i in x])
#               ]

# np.array([amp * np.exp(1j * 2 * pi * f * (i / fs)) for i in x])
# real portion
# real_I = [amp * 1 * np.cos(2 * pi * 1 * -f * (i / fs)) for i in x]
# # image portion
# imag_Q = [amp * np.sin(2 * pi * 1 * -f * (i / fs)) for i in x]

# real_I = cmplx_exp.real
# imag_Q = cmplx_exp.imag


# random
# y = [amp * np.random.randn() for i in x]
# y1 = [amp * np.random.randn() for i in x]

def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


# def flatten_list(l):
#     output = []
#     for i in l:
#         if type(i) == list:
#             flatten_list(i)
#         else:
#             output.append(i)
#
#     return output


class ScopeRect(object):
    def __init__(self, num_sigs, max_t=2 * T, dt=Ts):
        self.ax_r = ax_rect
        self.ax_r.grid(True)
        self.dt = dt
        self.max_t = max_t
        self.t_data = [0]
        self.y_data = []
        self.sig_lines_r = []
        for _ in range(num_sigs):
            self.sig_lines_r.append([self.ax_r.plot([], [], fstr)[0] for _, fstr in zip(range(2), ['', ''])])
            self.y_data.append([[0], [0]])

        # for lne in self.lines:
        #     self.ax_r.add_line(lne)

        self.ax_r.set_ylim(-amp - 2, amp + 2)
        self.ax_r.set_xlim(0, self.max_t)

    def update(self, emitted):

        if not pause:
            # print('rect values is ', y)
            last_t = self.t_data[-1]
            if continuous:
                if last_t > self.t_data[0] + self.max_t:
                    self.ax_r.set_xlim(last_t - self.max_t, last_t)

            t = self.t_data[-1] + self.dt
            self.t_data.append(t)

            for emitted_sig, sig_line_r, y_data_1 in zip(emitted, self.sig_lines_r, self.y_data):
                # for curr_y, curr_emitted, line in zip(self.y_data, emitted_sig, sig_line_r):
                #     curr_y.append(curr_emitted)
                #     line.set_data(self.t_data, curr_y)

                y_data_1[0].append(emitted_sig[0])
                y_data_1[1].append(emitted_sig[1])

                sig_line_r[0].set_data(self.t_data, y_data_1[0])
                sig_line_r[1].set_data(self.t_data, y_data_1[1])

        # print(list(flatten(self.sig_lines_r)))
        return list(flatten(self.sig_lines_r))


class ScopeRectCmbd(object):
    def __init__(self, num_sigs, max_t=2 * T, dt=Ts):
        self.ax_r = ax_rect_cmbd
        self.ax_r.grid(True)
        self.dt = dt
        self.max_t = max_t
        self.t_data = [0]
        self.y_data = []
        self.sig_lines_r = []
        for _ in range(num_sigs):
            self.sig_lines_r.append([self.ax_r.plot([], [], fstr)[0] for _, fstr in zip(range(2), ['', ''])])
            self.y_data.append([[0], [0]])

        self.sig_lines_r_cmbd = [self.ax_r.plot([], [], fstr, linewidth=lw,
                                                path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()])[0]
                                 for _, fstr, lw in
                                 zip(range(2), ['-.', '--'], [3, 2])]
        self.y_data_cmbd = [[0], [0]]

        # for lne in self.lines:
        #     self.ax_r.add_line(lne)

        self.ax_r.set_ylim(-amp - 2, amp + 2)
        self.ax_r.set_xlim(0, self.max_t)

    def update(self, emitted):

        if not pause:
            # print('rect values is ', y)
            last_t = self.t_data[-1]
            if continuous:
                if last_t > self.t_data[0] + self.max_t:
                    self.ax_r.set_xlim(last_t - self.max_t, last_t)

            t = self.t_data[-1] + self.dt
            self.t_data.append(t)

            for emitted_sig, sig_line_r, y_data_1 in zip(emitted, self.sig_lines_r, self.y_data):
                # for curr_y, curr_emitted, line in zip(self.y_data, emitted_sig, sig_line_r):
                #     curr_y.append(curr_emitted)
                #     line.set_data(self.t_data, curr_y)

                y_data_1[0].append(emitted_sig[0])
                y_data_1[1].append(emitted_sig[1])

                # sig_line_r[0].set_data(self.t_data, y_data_1[0])
                # sig_line_r[1].set_data(self.t_data, y_data_1[1])

        if not pause:

            real_list = []
            imag_list = []

            for emitted_sig in emitted:
                real_list.append(emitted_sig[0])
                imag_list.append(emitted_sig[1])

            real_sum = sum(real_list)
            imag_sum = sum(imag_list)

            self.y_data_cmbd[0].append(real_sum)
            self.y_data_cmbd[1].append(imag_sum)

            self.sig_lines_r_cmbd[0].set_data(self.t_data, self.y_data_cmbd[0])
            self.sig_lines_r_cmbd[1].set_data(self.t_data, self.y_data_cmbd[1])

        # print(list(flatten(self.sig_lines_r)))
        return list(flatten(self.sig_lines_r + self.sig_lines_r_cmbd))


class ScopePolar(object):
    def __init__(self, num_sigs):
        self.ax_p = ax_polar
        self.ax_p.set_rmax(1)
        self.mag_accu = []
        self.theta_accu = []
        self.sig_lines_p = []
        # self.lines = [plt.plot([], [])[0] for _ in range(2)]

        for _ in range(num_sigs):
            self.sig_lines_p.append(
                [self.ax_p.plot([], [], fstr, linewidth=lw)[0] for _, fstr, lw in
                 zip(range(4), ['-', '.', '-', '-'], [3, 1.5, 1.5, 1.5])])
            self.mag_accu.append([0])
            self.theta_accu.append([0])

    def update(self, emitted):
        # print('polar values is ', y)
        for emitted_sig, sig_line_p, mag_accu, theta_accu in zip(emitted, self.sig_lines_p, self.mag_accu,
                                                                 self.theta_accu):
            if not pause:

                input_num = complex(emitted_sig[0], emitted_sig[1])  # stored as 1+2j
                mag, theta = cm.polar(input_num)
                # print(theta)
                mag_accu.append(mag)
                theta_accu.append(theta)

                # adjust polar r limit
                if mag_accu[-1] > self.ax_p.get_rmax():
                    self.ax_p.set_rmax(mag_accu[-1] + 1)

                # clear mag and theta lists if one rotation is complete
                if theta_accu[-1] > 0 > theta_accu[-2]:
                    mag_accu.clear()
                    theta_accu.clear()
                    mag_accu.append(mag)
                    theta_accu.append(theta)

                cmplx = cm.rect(mag, theta)
                x = cmplx.real
                y = cmplx.imag
                # print(cmplx)

                mag_x, theta_x = cm.polar(complex(x, 0))
                mag_y, theta_y = cm.polar(complex(0, y))
                # print(mag_x, theta_x)
                # print(mag_y, theta_y)
                # rotating phasor
                sig_line_p[0].set_data([theta, theta], [0, mag])
                # phasor tracing
                # sig_line_p[1].set_data(theta_accu, mag_accu)
                # real tracing
                sig_line_p[2].set_data([theta_x, theta_x], [0, mag_x])
                # imag tracing
                sig_line_p[3].set_data([theta_y, theta_y], [0, mag_y])

        return list(flatten(self.sig_lines_p))


class ScopePolarCmbd(object):
    def __init__(self, num_sigs):
        self.ax_p = ax_polar_cmbd
        self.ax_p.set_rmax(1)
        self.mag_accu = []
        self.theta_accu = []
        self.sig_lines_p = []

        # self.lines = [plt.plot([], [])[0] for _ in range(2)]

        for _ in range(num_sigs):
            self.sig_lines_p.append(
                [self.ax_p.plot([], [], fstr, linewidth=lw)[0] for _, fstr, lw in
                 zip(range(4), [':', '.', '-', '-'], [3, 1.5, 1.5, 1.5])])
            self.mag_accu.append([0])
            self.theta_accu.append([0])

        self.sig_lines_p_cmbd = [self.ax_p.plot([], [], fstr, linewidth=lw)[0] for _, fstr, lw in
                                 zip(range(4), [':', '.', '-', '-'], [3, 1.5, 1.5, 1.5])]

    def update(self, emitted):
        # print('polar values is ', y)
        for emitted_sig, sig_line_p, mag_accu, theta_accu in zip(emitted, self.sig_lines_p, self.mag_accu,
                                                                 self.theta_accu):
            if not pause:

                input_num = complex(emitted_sig[0], emitted_sig[1])  # stored as 1+2j
                mag, theta = cm.polar(input_num)
                # print(theta)
                mag_accu.append(mag)
                theta_accu.append(theta)

                # adjust polar r limit
                if mag_accu[-1] > self.ax_p.get_rmax():
                    self.ax_p.set_rmax(mag_accu[-1] + 1)

                # clear mag and theta lists if one rotation is complete
                if theta_accu[-1] > 0 > theta_accu[-2]:
                    mag_accu.clear()
                    theta_accu.clear()
                    mag_accu.append(mag)
                    theta_accu.append(theta)

                cmplx = cm.rect(mag, theta)
                x = cmplx.real
                y = cmplx.imag
                # print(cmplx)

                mag_x, theta_x = cm.polar(complex(x, 0))
                mag_y, theta_y = cm.polar(complex(0, y))
                # print(mag_x, theta_x)
                # print(mag_y, theta_y)
                # rotating phasor
                sig_line_p[0].set_data([theta, theta], [0, mag])
                # phasor tracing
                # sig_line_p[1].set_data(theta_accu, mag_accu)
                # real tracing
                # sig_line_p[2].set_data([theta_x, theta_x], [0, mag_x])
                # imag tracing
                # sig_line_p[3].set_data([theta_y, theta_y], [0, mag_y])

        if not pause:

            real_list = []
            imag_list = []

            for emitted_sig in emitted:
                real_list.append(emitted_sig[0])
                imag_list.append(emitted_sig[1])

            real_sum = sum(real_list)
            imag_sum = sum(imag_list)

            mag, theta = cm.polar(complex(real_sum, imag_sum))

            # adjust polar r limit
            if mag > self.ax_p.get_rmax():
                self.ax_p.set_rmax(mag + 0.25)

            cmplx = cm.rect(mag, theta)
            x = cmplx.real
            y = cmplx.imag
            # print(cmplx)

            mag_x, theta_x = cm.polar(complex(x, 0))
            mag_y, theta_y = cm.polar(complex(0, y))
            # print(mag_x, theta_x)
            # print(mag_y, theta_y)
            # rotating phasor
            self.sig_lines_p_cmbd[0].set_data([theta, theta], [0, mag])
            # phasor tracing
            # self.sig_lines_p_cmbd[1].set_data(theta_accu, mag_accu)
            # real tracing
            self.sig_lines_p_cmbd[2].set_data([theta_x, theta_x], [0, mag_x])
            # imag tracing
            self.sig_lines_p_cmbd[3].set_data([theta_y, theta_y], [0, mag_y])

        return list(flatten(self.sig_lines_p + self.sig_lines_p_cmbd))


class Scope(ScopeRectCmbd, ScopePolarCmbd):
    def __init__(self, num_sigs, max_t=2 * T, dt=Ts):
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
                real = cmplx_exp.real[i]
                imag = cmplx_exp.imag[i]
                cmplx_exp_real_imag.append([real, imag])

            # real = real_I[i]
            # imag = imag_Q[i]
            i = i + 1

        # print(cmplx_exp_real_imag)
        yield cmplx_exp_real_imag


def onClick(event):
    global pause
    pause ^= True


fig = plt.figure(1, figsize=(10, 10))
ax_rect = plt.subplot(4, 1, 1)
ax_polar = plt.subplot(4, 1, 2, projection='polar')
ax_polar_cmbd = plt.subplot(4, 1, 3, projection='polar')
ax_rect_cmbd = plt.subplot(4, 1, 4)

# scope = ScopeRect(len(cmplx_exps))
scope_main = Scope(len(cmplx_exps))

# scope1 = ScopePolar(len(cmplx_exps))

interval = 10

fig.canvas.mpl_connect('key_press_event', onClick)

# pass a generator in "sineEmitter" to produce data for the update func
ani = animation.FuncAnimation(fig, scope_main.update, sine_emitter, interval=interval,
                              blit=True)

plt.show()
