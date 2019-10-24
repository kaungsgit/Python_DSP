import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.lines import Line2D
import matplotlib.animation as animation

# Your Parameters
amp = 1  # 1V        (Amplitude)
f = 1000  # 1kHz      (Frequency)
fs = 200000  # 200kHz    (Sample Rate)
T = 1 / f
Ts = 1 / fs

x_t = np.arange(0, fs * Ts, Ts)

# Select if you want to display the sine as a continuous wave
#  True = continuous (not able to zoom in x-direction)
#  False = Non-continuous  (able to zoom)
continuous = True

x = np.arange(fs)
y = [amp * np.cos(2 * np.pi * f * (i / fs)) for i in x]
y1 = [amp * np.sin(2 * np.pi * f * (i / fs)) for i in x]

plt.figure(1)
ax1 = plt.axes(xlim=(0, 2 * Ts * fs / f), ylim=(-1, 1))
line2d_1 = plt.plot(x_t, y)

plt.figure(2)
ax2 = plt.axes(xlim=(0, 2 * Ts * fs / f), ylim=(-1, 1))
line2d_2 = plt.plot(x_t, y1)


# shows 2 cycles
# plt.xlim((0, 2 * Ts * fs / f))


class Scope(object):
    def __init__(self, ax, maxt=2 * T, dt=Ts):
        self.ax = ax
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = [[0], [0]]
        # self.y1data = [0]
        # self.line = Line2D(self.tdata, self.ydata)
        # self.ax.add_line(self.line)
        self.lines = [plt.plot([], [])[0] for _ in range(2)]
        for lne in self.lines:
            self.ax.add_line(lne)

        self.ax.set_ylim(-amp - 2, amp + 2)
        self.ax.set_xlim(0, self.maxt)

    def update(self, y):
        lastt = self.tdata[-1]
        if continuous:
            if lastt > self.tdata[0] + self.maxt:
                self.ax.set_xlim(lastt - self.maxt, lastt)

        t = self.tdata[-1] + self.dt
        self.tdata.append(t)

        for y_val, curr_y_amp, lne in zip(self.ydata, y, self.lines):
            y_val.append(curr_y_amp)
            lne.set_data(self.tdata, y_val)
        # self.ydata.append(y[1])
        # self.lines[1].set_data(self.tdata, self.ydata)

        # print('a')
        return self.lines


def sineEmitter():
    for i in x:
        yield [y[i], y1[i]]


fig, ax = plt.subplots()
scope = Scope(ax)

# pass a generator in "sineEmitter" to produce data for the update func
ani = animation.FuncAnimation(fig, scope.update, sineEmitter, interval=10,
                              blit=True)

plt.show()
