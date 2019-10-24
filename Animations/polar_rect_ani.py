import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.lines import Line2D
import matplotlib.animation as animation

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
# real portion
real_I = [amp * 2 * np.cos(2 * pi * 2 * f * (i / fs)) for i in x]
# image portion
imag_Q = [amp * np.sin(2 * pi * 2 * f * (i / fs)) for i in x]


# random
# y = [amp * np.random.randn() for i in x]
# y1 = [amp * np.random.randn() for i in x]

# y = y + y1
#
# z = y + y1

# plt.figure(1)
# ax1 = plt.axes(xlim=(0, 2 * Ts * fs / f), ylim=(-1, 1))
# line2d_1 = plt.plot(x_t, y)
#
# plt.figure(2)
# ax2 = plt.axes(xlim=(0, 2 * Ts * fs / f), ylim=(-1, 1))
# line2d_2 = plt.plot(x_t, y1)


# shows 2 cycles
# plt.xlim((0, 2 * Ts * fs / f))


class ScopeRect(object):
    def __init__(self, ax_r, max_t=2 * T, dt=Ts):
        self.ax_r = ax_r
        self.dt = dt
        self.max_t = max_t
        self.t_data = [0]
        self.y_data = [[0], [0]]
        self.lines = [plt.plot([], [])[0] for _ in range(2)]

        for lne in self.lines:
            self.ax_r.add_line(lne)

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

            for curr_y, curr_emitted, line in zip(self.y_data, emitted, self.lines):
                curr_y.append(curr_emitted)
                line.set_data(self.t_data, curr_y)
        return self.lines


class ScopePolar(object):
    def __init__(self, ax_p, max_t=2 * T, dt=Ts):
        self.ax_p = ax_p
        self.dt = dt
        self.max_t = max_t
        self.t_data = [0]
        self.y_data = [[0], [0]]
        self.mag_accu = [0]
        self.theta_accu = [0]
        # self.lines = [plt.plot([], [])[0] for _ in range(2)]

        self.polars = [self.ax_p.plot([], [], sym)[0] for _, sym in zip(range(2), ['b-', 'b.'])]

    def update(self, emitted):
        # print('polar values is ', y)

        if not pause:
            mag = (emitted[0] ** 2 + emitted[1] ** 2) ** 0.5

            if emitted[0] >= 0 and emitted[1] >= 0:
                quadrant = 1
                theta = np.arctan(emitted[1] / emitted[0])
            elif emitted[0] < 0 <= emitted[1]:
                # 2nd quadrant
                quadrant = 2
                theta = pi + np.arctan(emitted[1] / emitted[0])
            elif emitted[0] < 0 and emitted[1] < 0:
                # 3rd quadrant
                quadrant = 3
                theta = pi + np.arctan(emitted[1] / emitted[0])
            else:
                # 4th quadrant
                quadrant = 4
                theta = 2 * pi + np.arctan(emitted[1] / emitted[0])
            # print(quadrant)
            # print(theta * 180 / pi, mag)

            self.mag_accu.append(mag)
            self.theta_accu.append(theta)

            # for curr_y, curr_emitted in zip(self.y_data, emitted):
            #     curr_y.append(curr_emitted)

            self.polars[0].set_data([theta, theta], [0, mag])
            self.polars[1].set_data(self.theta_accu, self.mag_accu)

        return self.polars


def sine_emitter():
    # for i in x:
    #     print('i is ', i)
    #     if not pause:
    #         yield [real_I[i], imag_Q[i]]
    #     else:
    #         yield [None, None]
    i = 0
    real = 0
    imag = 0
    while i < x.size:
        print('i is ', i)
        if not pause:
            real = real_I[i]
            imag = imag_Q[i]
            i = i + 1
        yield [real, imag]


def onClick(event):
    global pause
    pause ^= True


fig = plt.figure(1, figsize=(10, 10))
ax_rect = plt.subplot(1, 1, 1)
# fig, ax_rect = plt.subplots()
# ax_polar = plt.subplot(2, 1, 2, projection='polar')
scope = ScopeRect(ax_rect)

fig1 = plt.figure(2, figsize=(10, 10))
ax_polar = plt.subplot(1, 1, 1, projection='polar')
ax_polar.set_rmax(4)
# fig, ax_rect = plt.subplots()
# ax_polar = plt.subplot(2, 1, 2, projection='polar')
scope1 = ScopePolar(ax_polar)

interval = 10

fig.canvas.mpl_connect('button_press_event', onClick)
# fig1.canvas.mpl_connect('button_press_event', onClick)
# pass a generator in "sineEmitter" to produce data for the update func
ani = animation.FuncAnimation(fig, scope.update, sine_emitter, interval=interval,
                              blit=True)

ani1 = animation.FuncAnimation(fig1, scope1.update, sine_emitter, interval=interval,
                               blit=True)

# pause

plt.show()
