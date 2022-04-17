# import math
# print(math.factorial(20))

from scipy import stats
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np
from numpy.random import randn

# X = stats.binom(10, 0.2) # Declare X to be a binomial random variable
# X = stats.norm()
# print(X.pmf(3))           # P(X = 3)
# print(X.cdf(4))           # P(X <= 4)
# print(X.mean())           # E[X]
# print(X.var())            # Var(X)
# print(X.std())            # Std(X)
# print(X.rvs())            # Get a random sample from X
# print(X.rvs(10))          # Get 10 random samples form X

no_realz = 3
realz_len = 100000

x = np.zeros([no_realz, realz_len])
for i in range(no_realz):
    x[i, :] = randn(1, realz_len)

# plt.figure()
for i in range(no_realz):
    plt.subplot(no_realz, 1, i + 1)
    # plt.stem(x[i, :])
    histVal, _, _ = plt.hist(x[i, :], bins=200, density=True)
    plt.xlim(-5, 5)
    plt.ylim(0, max(histVal))

print(sum(x[:, 2]) / len(x[:, 2]))

# autocorrelation (correlation vs delay for same function)
# noise signal
N = 500
lag = np.arange(-N + 1, N)
noise = randn(N) + 1j * randn(N)

corr = sig.correlate(noise, noise)

plt.figure()
plt.plot(lag, np.abs(corr))
plt.xlabel("Lag [samples]")
plt.ylabel("Real Magnitude")
plt.title(f"AWGN Autocorrelation {N} Samples")

plt.show()
