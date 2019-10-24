import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np

N = 2
pi = np.pi
t = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(2 * pi * t)
y2 = np.cos(2 * pi * t)

dataframes = [pd.DataFrame({"x": np.sort(np.random.rand(100) * 100),
                            "y1": np.sin(t) * 20,
                            "y2": np.cos(t) * 20}) for _ in range(N)]

fig = plt.figure()
ax = plt.axes(xlim=(0, 100), ylim=(-30, 30))

lines = [plt.plot([], [])[0] for _ in range(2)]


def animate(i):
    lines[0].set_data(dataframes[i]["x"], dataframes[i]["y1"])
    lines[1].set_data(dataframes[i]["x"], dataframes[i]["y2"])
    return lines


anim = animation.FuncAnimation(fig, animate,
                               frames=N, interval=200, blit=True)

plt.show()



# import matplotlib.pyplot as plt
# import numpy as np
# import scipy as sp
# from matplotlib.lines import Line2D
# import matplotlib.animation as animation
#
# # Your Parameters
# amp = 1  # 1V        (Amplitude)
# f = 1000  # 1kHz      (Frequency)
# fs = 200000  # 200kHz    (Sample Rate)
# T = 1 / f
# Ts = 1 / fs
#
# x_t = np.arange(0, fs * Ts, Ts)
#
# # Select if you want to display the sine as a continuous wave
# #  True = continuous (not able to zoom in x-direction)
# #  False = Non-continuous  (able to zoom)
# continuous = False
#
# x = np.arange(fs)
# y = [amp * np.cos(2 * np.pi * f * (i / fs)) for i in x]
# y1 = [amp * np.sin(2 * np.pi * f * (i / fs)) for i in x]
#
# plt.figure(1)
# ax1 = plt.axes(xlim=(0, 2 * Ts * fs / f), ylim=(-1, 1))
# line2d_1 = plt.plot(x_t, y)
#
# plt.figure(2)
# ax2 = plt.axes(xlim=(0, 2 * Ts * fs / f), ylim=(-1, 1))
# line2d_2 = plt.plot(x_t, y1)
#
#
# # shows 2 cycles
# # plt.xlim((0, 2 * Ts * fs / f))
#
#
# # lines = [plt.plot([], [])[0] for _ in range(2)]
#
#
# class Scope(object):
#     def __init__(self, ax, maxt=2 * T, dt=Ts):
#         self.ax = ax
#         self.dt = dt
#         self.maxt = maxt
#         self.tdata = [0]
#         self.ydata = [0]
#         # self.line = Line2D(self.tdata, self.ydata)
#         self.lines = [plt.plot([], [])[0] for _ in range(2)]
#         # self.ax.add_line(self.line)
#
#         for lne in self.lines:
#             self.ax.add_line(lne)
#
#         self.ax.set_ylim(-amp, amp)
#         self.ax.set_xlim(0, self.maxt)
#
#     def update(self, y):
#         lastt = self.tdata[-1]
#         if continuous:
#             if lastt > self.tdata[0] + self.maxt:
#                 self.ax.set_xlim(lastt - self.maxt, lastt)
#
#         t = self.tdata[-1] + self.dt
#         self.tdata.append(t)
#
#         for y_amp, lne in zip(y, self.lines):
#             self.ydata.append(y_amp)
#             lne.set_data(self.tdata, self.ydata)
#         return tuple(self.lines)
#
#
# def sineEmitter():
#     for i in x:
#         yield [y[i], y1[i]]
#
#
# fig, ax = plt.subplots()
# scope = Scope(ax)
#
# # pass a generator in "sineEmitter" to produce data for the update func
# ani = animation.FuncAnimation(fig, scope.update, sineEmitter, interval=10,
#                               blit=True)
#
# plt.show()
