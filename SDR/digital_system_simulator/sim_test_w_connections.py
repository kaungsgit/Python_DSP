import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class Connector:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name
        self.connections = []
        self.value = 0
        self.ready = 0

    # def __str__(self):

    def connect(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        for input_ in inputs:
            self.connections.append(input_)

    def set(self, value):
        if self.value == value:
            return  # Ignore if no change
        self.value = value
        self.ready = 1
        for con in self.connections:
            con.set(value)


class Gain:
    def __init__(self, gain):
        self.inpath = Connector(self, 'A')
        self.outpath = Connector(self, 'B')
        self.gain = gain

    def compute(self):
        self.outpath.set(self.inpath.value * self.gain)


class Delay:
    def __init__(self):
        self.inpath = Connector(self, 'A')
        self.outpath = Connector(self, 'B')
        self.inval = 0
        self.outval = 0

    def compute(self):
        self.outpath.set(self.inpath.value)


class Sum:
    def __init__(self):
        self.inpath1 = Connector(self, 'A')
        self.inpath2 = Connector(self, 'B')
        self.outpath = Connector(self, 'C')

    def compute(self):
        self.outpath.set(self.inpath1.value + self.inpath2.value)


class Subtraction:
    def __init__(self):
        self.inpath1 = Connector(self, 'A')
        self.inpath2 = Connector(self, 'B')
        self.outpath = Connector(self, 'C')

    def compute(self):
        self.outpath.set(self.inpath1.value - self.inpath2.value)


x = [1, 2, 3, 4]
num_samples = 200

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

gain_in, gain_out, del_out = 0, 0, 0

Ts = 0.1
stop_time = num_samples * Ts
t = np.arange(0, stop_time, Ts)

# class ControlLoop:
#     def __init__(self):
#         gain1 = Gain(0.5)
#         delay1 = Delay()
#         delay2 = Delay()
#         sum1 = Sum()
#         sum2 = Sum()
#         gain2 = Gain(0.5 * Ts)
#         sub = Subtraction()
#
#         self.blocks = [gain1, delay1, delay2, sum1, sum2, gain2, sub]
#
gain1 = Gain(0.5)
delay1 = Delay()
delay2 = Delay()
sum1 = Sum()
sum2 = Sum()
gain2 = Gain(0.5 * Ts)
sub = Subtraction()

sub.outpath.connect([sum1.inpath1, gain1.inpath])
sum1.outpath.connect(([delay2.inpath, gain2.inpath]))
delay2.outpath.connect(sum1.inpath2)

gain1.outpath.connect(sum2.inpath1)
gain2.outpath.connect(sum2.inpath2)

sum2.outpath.connect(delay1.inpath)

delay1.outpath.connect(sub.inpath2)

delay1_out_arr = []
err = []
for i in range(num_samples):
    # gain1.inpath.set(i)
    sub.inpath1.set(1)

    delay1.compute()
    delay2.compute()

    sub.compute()
    gain1.compute()
    sum1.compute()
    gain2.compute()
    sum2.compute()

    # delay1.inpath = gain1.outpath

    delay1_out_arr.append(delay1.outpath.value)
    err.append(sub.outpath.value)

plt.plot(t, delay1_out_arr, '-o', label='delay_out')
plt.plot(t, err, '-o', label='error')

# plt.plot(x, '-o', label='input')
plt.legend()
plt.show()

# sub.outpath.connect(sum1.inpath1)
# sum1.outpath.connect(delay2.inpath)
# delay2.outpath.connect(sum1.inpath2)
# sum1.outpath.connect(gain2.inpath)
# gain2.outpath.connect(sum2.inpath1)
#
# sub.outpath.connect(sum1.inpath1)
# gain1.outpath.connect(sum2.inpath2)
#
# gain2.outpath.connect(delay1.inpath)
#
# # gain1.outpath.connect(delay1.inpath)
#
# sub.outpath.connect(gain1.inpath)
# # sub.inpath2.connect(delay1.outpath) # as of now, only outpath.connect(inpath) works,
# # because compute changes only outpath, which in turn changes inpath.
# delay1.outpath.connect(sub.inpath2)
