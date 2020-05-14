import numpy as np
from collections import OrderedDict

# to compute all possible permutations
# using itertools.product()
import itertools


# # initializing list of list
# all_list = [[1, 3], [6], [8, 10]]
#
# # printing lists
# print("The original lists are : " + str(all_list))
#
# # using itertools.product()
# # to compute all possible permutations
# res = list(itertools.product(*all_list))
#
# loop_param = OrderedDict()
# loop_param['temps'] = [-10, 15]
# loop_param['Fs_list'] = [4000, 6000]
# loop_param['fins'] = [4500, 2234]
#
# curr_param = OrderedDict()
#
# b = np.zeros(len(loop_param))
# for i, val in enumerate(loop_param.values()):
#     b[i] = len(val)
#
# c = np.prod(b)
#
# print(f'Number of iteration is {c}')
#
# # for key, value in loop_param.items():
# #     print(key)
# #     print(value)
# #     for list_val in value:
# #         print(list_val)
#
# loop_param_keys = list(loop_param.keys())
#
# for temp in loop_param['temps']:
#
#     n = loop_param_keys.index('temps')
#     print(f'Current {loop_param_keys[n]} is {temp}')
#     curr_param[loop_param_keys[n]] = temp
#
#     for fs in loop_param['Fs_list']:
#
#         n = loop_param_keys.index('Fs_list')
#         print(f'Current {loop_param_keys[n]} is {fs}')
#         curr_param[loop_param_keys[n]] = fs
#
#         for fin in loop_param['fins']:
#
#             n = loop_param_keys.index('fins')
#             print(f'Current {loop_param_keys[n]} is {fin}')
#             curr_param[loop_param_keys[n]] = fin
#
#             # inside the innermost for loop
#             # make up string with all the curr param keys and values
#             name2 = ''
#             for key, value in curr_param.items():
#                 name1 = key + str(value) + '_'
#                 name2 = name2 + name1
#
#             print(name2)
#
# print('1')


def for_loop(loop_dict_iter):
    for x in next(loop_dict_iter):
        print(x)

        # if next(loop_dict_iter) is 'Done':
        #     pass
        # else:

        try:
            for_loop(loop_dict_iter)
        except StopIteration:
            pass

    return 0


# for_loop(iter(loop_param.values()))
#
# mylist = iter(["apple", "banana", "cherry"])
# x = next(mylist, "orange")
# print(x)
# x = next(mylist, "orange")
# print(x)
# x = next(mylist, "orange")
# print(x)
# x = next(mylist, "orange")
# print(x)


# temp1 = Temp('temp1')
# temp1.set_param('temp', -10)
#
# Fadc1 = Fadc('Fadc1')
#
# Fadc1.set_param('fadc', 1500)

''' **************** child class changing parent's class attribute that also affects other child classes ************'''


class Foo(object):
    data = "abc"


class Bar(Foo):
    Foo.data += "def"


class Bar1(Foo):
    def print_data(self):
        print(Foo.data)


b = Bar()
b.data

c = Bar1()
c.data
''' **************** child class changing parent's class attribute that also affects other child classes ************'''

''' This is going to be the backbone of my generic_sweep.py. No more growing nested for loops when new param is added!
    To add more sweep params, add more subclasses to GenericParam and modify the set_param method for how this 
    specifically param needs to be set.
    '''


class GenericParams:
    shared_vars = dict()
    curr_params = dict()

    def __init__(self, name='generic_param'):
        self.key = name
        self.value = -1000

    def set_param(self, key=None, value=None):
        print(f'Setting {key} to {value}...')
        self.value = value
        self.curr_params[key] = value


class Temp(GenericParams):

    def set_param(self, key=None, value=None):
        super().set_param(key, value)
        if value == -10:
            # set some class attribute that other child class will use in its set_param method
            GenericParams.shared_vars['brd'] = 10
            GenericParams.shared_vars['brd5'] = 1010
            print('Class attri 2 is changed in setting Temp')
        else:
            GenericParams.shared_vars['brd'] = -1

        print('end of set_param')


class Fadc(GenericParams):

    def set_param(self, key=None, value=None):
        super().set_param(key, value)
        if GenericParams.shared_vars['brd1'] == 10:
            print('doing some extra stuff in setting Fadc')


class Fin(GenericParams):

    def set_param(self, key=None, value=None):
        super().set_param(key, value)
        if GenericParams.shared_vars['brd5'] == 1010:
            print('doing some extra stuff in setting Fin')


# add more loop params here
# the loop param objects created must have the same name as the key entered here
loop_param = OrderedDict()
loop_param['Temp'] = [-10, 15]
loop_param['Fadc'] = [4000, 6000]
loop_param['Fin'] = [4500, 2234]

keys, values = zip(*loop_param.items())
iterations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# set some parent class attribute that one or more child class will need in their set_param method

GenericParams.shared_vars['brd'] = 1
GenericParams.shared_vars['brd1'] = 2
GenericParams.shared_vars['brd2'] = 3

for i in iterations:
    print(i)

    for param_key, param_value in i.items():
        # print(f'param_key is {param_key} and param_value is {param_value}')

        # print(f'temp is {i["Temp"]}')

        # create the class name from param_key
        curr_param_class = globals()[param_key]
        # create class instance from param_key
        curr_param_class_instance = curr_param_class(name=param_key)

        curr_param_class_instance.set_param(param_key, param_value)

    print('All params are set. Ready to collect data!!!')

print('1')
