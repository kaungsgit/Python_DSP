import numpy as np
from collections import OrderedDict

loop_param = OrderedDict()
loop_param['temps'] = [-10, 15]
loop_param['Fs_list'] = [4000, 6000]
loop_param['fins'] = [4500, 2234]

curr_param = OrderedDict()

b = np.zeros(len(loop_param))
for i, val in enumerate(loop_param.values()):
    b[i] = len(val)

c = np.prod(b)

print(f'Number of iteration is {c}')

# for key, value in loop_param.items():
#     print(key)
#     print(value)
#     for list_val in value:
#         print(list_val)

loop_param_keys = list(loop_param.keys())

for temp in loop_param['temps']:

    n = loop_param_keys.index('temps')
    print(f'Current {loop_param_keys[n]} is {temp}')
    curr_param[loop_param_keys[n]] = temp

    for fs in loop_param['Fs_list']:

        n = loop_param_keys.index('Fs_list')
        print(f'Current {loop_param_keys[n]} is {fs}')
        curr_param[loop_param_keys[n]] = fs

        for fin in loop_param['fins']:

            n = loop_param_keys.index('fins')
            print(f'Current {loop_param_keys[n]} is {fin}')
            curr_param[loop_param_keys[n]] = fin

            # inside the innermost for loop
            # make up string with all the curr param keys and values
            name2 = ''
            for key, value in curr_param.items():
                name1 = key + str(value) + '_'
                name2 = name2 + name1

            print(name2)

print('1')


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
