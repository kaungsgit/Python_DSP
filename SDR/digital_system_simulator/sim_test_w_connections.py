from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
from abc import ABC, abstractmethod


class Connector:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name
        self.connections = []
        self.value = 0
        self.ready = 0

    # def __str__(self):

    def connect(self, inputs: Union[Connector, List[Connector]]):
        if not isinstance(inputs, list):
            inputs = [inputs]
        for input_ in inputs:
            self.connections.append(input_)

    def set(self, value):
        if self.value == value and self.value != 0:
            return  # Ignore if no change
        self.value = value
        self.ready = 1

        for con in self.connections:
            con.set(value)


class Block:
    count = 0

    def __init__(self, name=None):
        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__ + str(self.__class__.count)
            self.__class__.count += 1


class Block2(Block):
    def __init__(self, name=None):
        super().__init__(name)
        self.inpath = Connector(self, 'A')
        self.outpath = Connector(self, 'B')


class Block3(Block):
    def __init__(self, name=None):
        super().__init__(name)
        self.inpath1 = Connector(self, 'A')
        self.inpath2 = Connector(self, 'B')
        self.outpath = Connector(self, 'C')


class Gain(Block2):

    def __init__(self, gain, name=None):
        super().__init__(name)
        # self.inpath = Connector(self, 'A')
        # self.outpath = Connector(self, 'B')
        self.gain = gain

    def compute(self):
        self.outpath.set(self.inpath.value * self.gain)


class Delay(Block2):
    def __init__(self, name=None):
        super().__init__(name)

        # self.inpath = Connector(self, 'A')
        # self.outpath = Connector(self, 'B')
        # self.inval = 0
        # self.outval = 0

    def compute(self):
        self.outpath.set(self.inpath.value)


class Sum(Block3):
    def __init__(self, name=None):
        super().__init__(name)

        # self.inpath1 = Connector(self, 'A')
        # self.inpath2 = Connector(self, 'B')
        # self.outpath = Connector(self, 'C')

    def compute(self):
        self.outpath.set(self.inpath1.value + self.inpath2.value)


class Subtraction(Block3):
    def __init__(self, name=None):
        super().__init__(name)
        # self.inpath1 = Connector(self, 'A')
        # self.inpath2 = Connector(self, 'B')
        # self.outpath = Connector(self, 'C')

    def compute(self):
        self.outpath.set(self.inpath1.value - self.inpath2.value)


num_samples = 200
x = np.ones(num_samples)

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


class ControlLoop(ABC):

    def __init__(self):
        self.input_node = None
        self.output_node = None
        self.error_node = None

        self.specify_structure()

        self.blocks = [getattr(self, i) for i in self.__dir__() if i[:2] != '__' if isinstance(getattr(self, i), Block)]

    @abstractmethod
    def specify_structure(self):
        pass

    def create_run_order(self):

        self.input_node.ready = 1  # input is always ready

        delay_blocks = [i for i in self.blocks if isinstance(i, Delay)]
        # for count, d_block in enumerate(delay_blocks):
        #     if delay_blocks[count+1] in d_block.outpath.connections:
        for d_block in delay_blocks:  # @todo: re-order delay blocks if they are daisy-chained.
            d_block.compute()

        blocks_wo_delays = list(filter(lambda i: i not in delay_blocks, self.blocks))
        num_blocks_wo_delays = len(blocks_wo_delays)

        blocks_wo_delays_copy = blocks_wo_delays  # this = is just referencing, it's not copying.
        executed_blocks = []
        while len(executed_blocks) < num_blocks_wo_delays:
            for block in blocks_wo_delays_copy:
                if isinstance(block, Block2):
                    if block.inpath.ready:
                        block.compute()
                        executed_blocks.append(block)

                elif isinstance(block, Block3):
                    if block.inpath1.ready and block.inpath2.ready:
                        block.compute()
                        executed_blocks.append(block)

            for i in executed_blocks:
                if i in blocks_wo_delays_copy:
                    blocks_wo_delays_copy.remove(i)

        self.execution_order = delay_blocks + executed_blocks

    def run(self, input_data):

        self.create_run_order()

        delay1_out_arr = []
        err = []
        for i in input_data:
            self.input_node.set(i)

            for j in self.execution_order:
                j.compute()

            delay1_out_arr.append(self.output_node.value)
            err.append(self.error_node.value)

        return delay1_out_arr, err


class Type1CLExample(ControlLoop):

    def specify_structure(self):
        # specify blocks
        self.gain1 = Gain(0.5)
        self.delay1 = Delay()
        self.delay2 = Delay()
        self.sum1 = Sum()
        self.sum2 = Sum()
        self.gain2 = Gain(0.5 * Ts)
        self.sub = Subtraction()

        # specify connections
        self.sub.outpath.connect([self.sum1.inpath1, self.gain1.inpath])
        self.sum1.outpath.connect(([self.delay2.inpath, self.gain2.inpath]))
        self.delay2.outpath.connect(self.sum1.inpath2)

        self.gain1.outpath.connect(self.sum2.inpath1)
        self.gain2.outpath.connect(self.sum2.inpath2)

        self.sum2.outpath.connect(self.delay1.inpath)

        self.delay1.outpath.connect(self.sub.inpath2)

        # specify input, output, and error nodes
        self.input_node = self.sub.inpath1
        self.output_node = self.delay1.outpath
        self.error_node = self.sub.outpath


class Type0CLExample(ControlLoop):

    def specify_structure(self):
        # specify blocks
        self.gain1 = Gain(0.5)
        self.delay1 = Delay()
        # self.delay2 = Delay()
        # self.sum1 = Sum()
        # self.sum2 = Sum()
        # self.gain2 = Gain(0.5 * Ts)
        self.sub = Subtraction()

        # specify connections
        self.sub.outpath.connect(self.gain1.inpath)
        self.gain1.outpath.connect(self.delay1.inpath)
        self.delay1.outpath.connect(self.sub.inpath2)

        # specify input, output, and error nodes
        self.input_node = self.sub.inpath1
        self.output_node = self.delay1.outpath
        self.error_node = self.sub.outpath


t1_cl = Type1CLExample()
delay1_out_arr, err = t1_cl.run(x)

plt.figure()
plt.plot(t, delay1_out_arr, '-o', label='delay_out')
plt.plot(t, err, '-o', label='error')
plt.title('Type1 Control Loop')
plt.legend()

t0_cl = Type0CLExample()
delay1_out_arr, err = t0_cl.run(x)
plt.figure()
plt.plot(t, delay1_out_arr, '-o', label='delay_out')
plt.plot(t, err, '-o', label='error')
plt.title('Type0 Control Loop')
plt.legend()

plt.show()

# # *********************************** Sim without ControlLoop class  ***********************************
# gain1 = Gain(0.5)
# delay1 = Delay()
# delay2 = Delay()
# sum1 = Sum()
# sum2 = Sum()
# gain2 = Gain(0.5 * Ts)
# sub = Subtraction()
#
# sub.outpath.connect([sum1.inpath1, gain1.inpath])
# sum1.outpath.connect(([delay2.inpath, gain2.inpath]))
# delay2.outpath.connect(sum1.inpath2)
#
# gain1.outpath.connect(sum2.inpath1)
# gain2.outpath.connect(sum2.inpath2)
#
# sum2.outpath.connect(delay1.inpath)
#
# delay1.outpath.connect(sub.inpath2)
#
# delay1_out_arr = []
# err = []
# for i in range(num_samples):
#     # gain1.inpath.set(i)
#     sub.inpath1.set(1)
#
#     delay1.compute()
#     delay2.compute()
#
#     sub.compute()
#     gain1.compute()
#     sum1.compute()
#     gain2.compute()
#     sum2.compute()
#
#     # delay1.inpath = gain1.outpath
#
#     delay1_out_arr.append(delay1.outpath.value)
#     err.append(sub.outpath.value)
# plt.figure()
# plt.plot(t, delay1_out_arr, '-o', label='delay_out')
# plt.plot(t, err, '-o', label='error')
#
# plt.legend()
# plt.show()

# # *********************************** Sim without ControlLoop class  ***********************************


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
