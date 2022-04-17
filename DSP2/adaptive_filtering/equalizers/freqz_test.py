import komm
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import sys
import pprint as pp
import numpy.random as random

# sys.path.append("../")
import custom_tools.fftplot as fftplot
from scipy.fftpack import fft, fftshift, fftfreq, ifft
import custom_tools.handyfuncs as hf

b_k = [0.1, -0.1, 0.05, 1, 0.05]
b_k = [1, 1, 1]
a_k = [1, 1]
w, h = sig.freqz(b_k, a_k)
plt.figure()
plt.plot(w, hf.db(h))
plt.title('Raised Cosine Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid()

b_k = [0.1, -0.1, 0.05, 1, 0.05]
w, h = sig.freqz(b_k)
plt.figure()
plt.plot(w, hf.db(h))
plt.title('Raised Cosine Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid()

plt.show()
