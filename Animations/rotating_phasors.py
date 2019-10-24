import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# y_polar = np.zeros(100)
# amp = np.linspace(0, 1, 100)
#
# r = np.arange(0, 2, 0.01)
# theta = 2 * np.pi * r
#
# plt.figure(1)
# plt.plot(r, theta)
#
# plt.figure(2)
# ax = plt.subplot(111, projection='polar')
# ax.plot(theta, r)
# ax.set_rmax(2)
# ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
# ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
# ax.grid(True)
#
# ax.set_title("A line plot on a polar axis", va='bottom')
# plt.show()


# plt.figure(1)
# theta = [np.pi / 4, np.pi / 4]
# r = [0, 1]
# ax = plt.subplot(111, projection='polar')
# # ax.plot(theta, r, color='b')
# # ax.plot(np.pi/4, 1, '>', color='b')
#
# # theta, r
#
# # plt.arrow(0, 0, np.pi/2, 1, head_width = 0.1, head_length=0.1)
# ax.set_rmax(2.0)
# arr1 = plt.arrow(0, 0, 0, 1, alpha=0.5, width=0.05,
#                  edgecolor='black', facecolor='blue', lw=2, zorder=5)
#
# # arrow at 45 degree
# arr2 = plt.arrow(45 / 180. * np.pi, 0, 0, 1, alpha=0.5, width=0.05,
#                  edgecolor='black', facecolor='green', lw=2, zorder=5)


# plt.figure(2)
# plt.polar(thetas, rs)


pi = np.pi
thetas = np.linspace(0, 20 * pi, 1000)
rs = 1 * np.ones(100)
# rs = 2 * np.linspace(0, 1, 100)

thetas_1 = np.linspace(0, 40 * pi, 1000)
rs_1 = 0.5 * np.ones(100)

def wave_gen():
    for theta, r, theta_1, r_1 in zip(thetas, rs, thetas_1, rs_1):
        yield [theta, r, theta_1, r_1]


fig = plt.figure(3)
ax = plt.subplot(111, projection='polar')

polars = [ax.plot([], [])[0] for _ in range(2)]

# polar_plot = ax.plot(thetas, rs)
# polar_plot[0].set_data(thetas, rs * 0.5)


def animate(data):
    ax.clear()

    polars[0].set_data([data[0], data[0]], [0, data[1]])
    polars[1].set_data([data[2], data[2]], [0, data[3]])

    # obj = ax.plot([data[0], data[0]], [0, data[1]], color='blue')
    # obj1 = ax.plot([data[2], data[2]], [0, data[3]], color='black')
    ax.set_rmax(2)
    # arr2 = ax.arrow(theta, 0, 0, 1, alpha=0.5, width=0.05,
    #                  edgecolor='black', facecolor='green', lw=2, zorder=5)
    # return [obj, obj1]
    return polars


# fig = plt.figure(4)

ani = animation.FuncAnimation(fig, animate, wave_gen, interval=10,
                              blit=True)

plt.show()
