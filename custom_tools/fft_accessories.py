import numpy as np


def fftfreq_RLyonBook(N, d):
    """ This function is different from fftfreq from scipy
    in that it returns the first N/2+1 plus the second half (N/2-1 in scipy edition) as fftfreq for even N
    for odd N, it returns the first (N+1)/2 plus the second half ((N-1/)2 in scipy edition)
    (this is how Richard Lyon's DSP book states)
    """
    if N % 2 == 0:
        # even
        a1 = np.arange(0, N / 2 + 1, 1)
        a2 = np.arange(-N / 2 + 1, 0, 1)
        return np.concatenate((a1, a2)) / (N * d)
    else:
        # odd
        a1 = np.arange(0, (N + 1) / 2, 1)
        a2 = np.arange(-(N - 1) / 2, 0, 1)
        return np.concatenate((a1, a2)) / (N * d)


def angle2(x):
    """ Angle calculation to avoid Python numpy weird issue for angle function
    np.angle(-0 - 0j) = 0.0 (desired result)
    np.angle(-0.0 - 0.0j) = 3.14 (float numbers are IEEE754, not desired for angle calculation).
    Solution is to convert from float to int for float zero values (0.0)
    """
    res_arr = []

    for i in x:
        imag = i.imag
        real = i.real

        if real == 0 and isinstance(real, float):
            real = 0

        if imag == 0 and isinstance(real, float):
            imag = 0

        res = np.arctan2(imag, real)

        res_arr.append(res)

    return np.array(res_arr)
