import numpy as np
import matplotlib.pyplot as plt


def db(x):
    # returns dB of number and avoids divide by 0 warnings
    x = np.array(x)
    x_safe = np.where(x == 0, 1e-7, x)
    return 20 * np.log10(np.abs(x_safe))


def responsePlot(w, h, title):
    plt.subplot(2, 1, 1)
    plt.semilogx(w / (2 * np.pi), 20 * np.log10(np.abs(h)))
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.title(title)
    plt.subplot(2, 1, 2)
    plt.semilogx(w / (2 * np.pi), np.unwrap(np.angle(h)) * 360 / (2 * np.pi))
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Angle [deg]')
    # plt.show()


def return_sci_notation(x):
    x = format(x, 'e')
    num, pow10 = x.split('e')

    num = float(num)

    pow10 = float(pow10)

    return num, pow10
