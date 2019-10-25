"""
@author: Kaung Myat San Oo
Animation for understanding rotating phasors, negative frequency in DSP concepts

Add, remove, or modify signals in cmplx_exps
Script will animate the individual phasors and the combined output in both rectangular and polar plots
Press any keyboard key to pause the animation

"""

# @todo
# remove complex arrays and use complex variable type
# find out why the update is slow
# reorganize the object oriented scheme

import matplotlib.pyplot as plt
import numpy as np
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

phi = 0

cmplx_exps = [np.ones(fs),
              np.array([amp * np.exp(1j * (2 * pi * f * (i / fs) + phi)) for i in x]),
              np.array([amp * 1 * np.exp(1j * (2 * pi * (-f / 1) * (i / fs) + phi)) for i in x])
              ]


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
        self.t_data = [0]
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
                                 zip(range(2), ['-.', '--'], [3, 2])]
        self.y_data_cmbd = [[0], [0]]

        # adding legend
        self.sig_lines_r_cmbd[0].set_label('In-phase or Real')
        self.sig_lines_r_cmbd[1].set_label('Quadrature or Imag')
        self.ax_r.legend()

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
            self.t_data.append(t)

            for emitted_sig, sig_line_r, y_data_1 in zip(emitted, self.sig_lines_r, self.y_data):
                y_data_1[0].append(emitted_sig[0])
                y_data_1[1].append(emitted_sig[1])

                # these will show the individual line imag and real parts of each signal in cmplx_exps
                # these are usually commented out to avoid clutter
                # sig_line_r[0].set_data(self.t_data, y_data_1[0])
                # sig_line_r[1].set_data(self.t_data, y_data_1[1])

        # combined output drawing
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

            # this will draw the final combined output of the signals in cmplx_exps
            self.sig_lines_r_cmbd[0].set_data(self.t_data, self.y_data_cmbd[0])
            self.sig_lines_r_cmbd[1].set_data(self.t_data, self.y_data_cmbd[1])

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
                 zip(range(4), [':', '.', '-', '-'], [3, 1.5, 1.5, 1.5])])
            self.mag_accu.append([0])
            self.theta_accu.append([0])

        # data lines for drawing the real and imag time waves of combined signals in cmplx_exps
        self.sig_lines_p_cmbd = [self.ax_p.plot([], [], fstr, linewidth=lw)[0] for _, fstr, lw in
                                 zip(range(4), [':', '.', '-', '-'], [3, 1.5, 1.5, 1.5])]

    def update(self, emitted):
        # drawing the individual signals
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

                mag_x, theta_x = cm.polar(complex(x, 0))
                mag_y, theta_y = cm.polar(complex(0, y))

                # rotating phasor
                sig_line_p[0].set_data([theta, theta], [0, mag])

                # phasor edge tracing
                # sig_line_p[1].set_data(theta_accu, mag_accu)

                # these will draw the real and imag component of each signal in cmplx_exps on the polar plot
                # usually commented out to avoid clutter
                # # projection to real tracing
                # sig_line_p[2].set_data([theta_x, theta_x], [0, mag_x])
                # # projection to imag tracing
                # sig_line_p[3].set_data([theta_y, theta_y], [0, mag_y])

        # drawing of combined output
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

            i = i + 1

        yield cmplx_exp_real_imag


def onClick(event):
    global pause
    pause ^= True


fig = plt.figure(1, figsize=(10, 10))

ax_polar_cmbd = plt.subplot(2, 1, 1, projection='polar')

ax_rect_cmbd = plt.subplot(2, 1, 2)

scope_main = Scope(len(cmplx_exps))

interval = 10

fig.canvas.mpl_connect('key_press_event', onClick)

# pass a generator in "sineEmitter" to produce data for the update func
ani = animation.FuncAnimation(fig, scope_main.update, sine_emitter, interval=interval,
                              blit=True)

plt.show()
