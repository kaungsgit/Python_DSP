import csv
import matplotlib.pyplot as plt
import numpy as np
import custom_tools.fftplot as fftplot

insert_loss = []
freq = []
with open('./Data/insertion_loss_cable.csv') as f:
    csv_reader = csv.reader(f, delimiter=',')

    for row in csv_reader:
        # print(row)
        freq.append(float(row[0]))
        insert_loss.append(float(row[1]))

n = len(insert_loss)
freq = np.array(freq)
insert_loss = np.array(insert_loss)

plt.figure()
plt.plot(freq, insert_loss)
plt.xlabel('Freq (Hz)')
plt.ylabel('Amplitude (dB)')

freq_step = np.mean(np.gradient(freq))
print('freq_step is {}MHz'.format(freq_step / 1e6))
# need 25MHz freq_step, so 17/10*25 = 42.5
fs = 1 / freq_step

plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(insert_loss, fs=fs), drange=120)

interp = 17
insert_loss_1 = np.zeros(len(insert_loss) * interp)
insert_loss_1[::interp] = insert_loss

plt.figure()
fftplot.plot_spectrum(*fftplot.winfft(insert_loss_1, fs=fs), drange=120)

# interpolation filter


plt.show()
