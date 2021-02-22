import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np

N = 1
pi = np.pi
t = np.linspace(0, 2 * np.pi, 10)
dt = 2 * pi / 10
y1 = np.sin(2 * pi * t)
y2 = np.cos(2 * pi * t)

dataframes = [pd.DataFrame({"x": np.linspace(0, 2 * np.pi, 10),
                            "y1": np.sin(t) * 20,
                            "y2": np.cos(t) * 20}) for _ in range(N)]

fig = plt.figure()
ax = plt.axes(xlim=(0, 10 * pi), ylim=(-30, 30))

lines = [plt.plot([], [])[0] for _ in range(2)]

t_data = [0]
y1_data = [0]
y2_data = [0]


def init():
    ax.clear()
    ax.set_xlim(0, 10 * np.pi)
    ax.set_ylim(-30, 30)
    return lines


def animate(i):
    # lines[0].set_data(dataframes[i]["x"], dataframes[i]["y1"])
    # lines[1].set_data(dataframes[i]["x"], dataframes[i]["y2"])

    t_data.append(t_data[-1]+dt)
    y1_data.append(dataframes[0]['y1'][i])
    y2_data.append(dataframes[0]['y2'][i])

    lines[0].set_data(t_data, y1_data)
    lines[1].set_data(t_data, y2_data)

    return lines


def gen_frames():
    frames = np.arange(0, 10)
    for i in frames:
        yield i


anim = animation.FuncAnimation(fig, animate,
                               gen_frames, interval=20, blit=True, init_func=init)

plt.show()
