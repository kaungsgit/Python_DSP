import numpy as np


def calc_coherent_freq(fin, fs, N):
    """
    fS is the sampling frequency
    fIN is the input frequency
    M is the integer number of cycles in the data record
    N is the integer, factor of 2, number of samples in the record
    """

    M = np.floor(fin / fs * N)
    if M % 2 == 0:
        print('M is even')
        M = M + 1  # M must be odd

    if M < 1:
        print('Cant calculate f_coherent')
        f_coherent = fin
    else:
        f_coherent = M / N * fs

    return f_coherent, M
