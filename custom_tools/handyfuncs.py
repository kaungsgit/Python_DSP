import numpy as np


def db(x):
    # returns dB of number and avoids divide by 0 warnings
    x = np.array(x)
    x_safe = np.where(x == 0, 1e-7, x)
    return 20 * np.log10(np.abs(x_safe))