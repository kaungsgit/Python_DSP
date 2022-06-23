import numpy as np
import matplotlib.pyplot as plt

num_samples = 200
x = np.ones(num_samples)
Ts = 0.1
stop_time = num_samples * Ts
t = np.arange(0, stop_time, Ts)

kp = 0.5
ki = 0
err_i = 0
err_p = 0
y = np.zeros(num_samples)
post_lf = np.zeros(num_samples)
err = np.zeros(num_samples)

for i in range(num_samples):

    err[i] = (x[i] - y[i])
    err_i += err[i] * Ts
    err_p = err[i]

    post_lf[i] = err_p * kp + err_i * ki

    if i < num_samples - 1:
        y[i + 1] = post_lf[i]

plt.figure()
plt.plot(t, y, '-o', label='y')
plt.plot(t, err, '-o', label='err')

plt.legend()

input1 = 1
y_pre = 0
error = 0
out_array = []
error_array = []
# Dan's way of modelling
for i in range(num_samples):
    y_out = y_pre

    y_pre = error * kp
    error = input1 - y_out

    error_array.append(error)
    out_array.append(y_out)

plt.figure()
plt.plot(t, out_array, '-o', label='y')
plt.plot(t, error_array, '-o', label='err')
plt.legend()
plt.show()
