import numpy as np
import matplotlib.pyplot as plt


class Gain:
    def __init__(self, name, gain):
        self.name = name
        self.gain = gain
        self.inval = 0
        self.outval = 0

    def compute(self, inval):
        self.inval = inval
        self.outval = inval * self.gain
        return self.outval


class Delay:
    def __init__(self, name, delay):
        self.name = name
        self.delay = delay
        self.counter = 0
        self.inval = 0
        self.outval = 0
        self.array = np.zeros(delay + 1)

    def compute(self, inval):
        # self.array[0] = inval
        # self.array[1] = self.array[0]

        self.inval = inval
        # if self.counter == self.delay:
        #     self.outval = self.inval
        # else:
        #     self.counter += 1
        #     pass

        self.outval = self.inval

        return self.outval


x = [1, 2, 3, 4]

# study how to model neural networks from scratch
# this should be similar.
# http://openbookproject.net/courses/python4fun/logic.html
# https://virantha.com/2017/09/22/hardware-simulation-using-curio/
# https://simupy.readthedocs.io/en/latest/overview.html
# https://python.plainenglish.io/how-to-build-simulation-models-with-python-219b33ce9625
# https://www.youtube.com/watch?v=Os7ppbJh4To
# https://pypi.org/project/bdsim/

# simulink alternative
# https://www.collimator.ai/solutions/control-systems

gain1 = Gain('gain1', 1)
delay1 = Delay('delay1', 1)

# gain1 = Gain(gain_in, gain_out, 1)
# delay1 = Delay(gain_out, del_out)

pre_delay = 0
delay1_out_arr = []
for i in x:
    delay1_out = delay1.compute(pre_delay)

    gain1_out = gain1.compute(i)
    pre_delay = gain1_out
    delay1_out_arr.append(delay1_out)
    # print(i)

plt.plot(delay1_out_arr, '-o', label='delay_out')
plt.plot(x, '-o', label='input')
plt.legend()
plt.show()
