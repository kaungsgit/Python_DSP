import numpy as np
from collections import OrderedDict
import globals as swp_gbl
# to compute all possible permutations
# using itertools.product()
import itertools
import parameter_classes as prm_cls
import datalogger
import setupFiles.Jira_1856_test_setup as swpSetup

loop_param = OrderedDict()

dut_name = swpSetup.dut_name
board_name = swpSetup.board_name
JIRA_task_no = swpSetup.JIRA_task_no
misc_tag = 'Fs_Fin_Sweep'

for key, value in swpSetup.loop_param.items():
    loop_param[key] = value

keys, values = zip(*loop_param.items())
iterations = [OrderedDict(zip(keys, v)) for v in itertools.product(*values)]

for i in iterations:
    # print('______Current parameters are ', i)

    # this makes the set_param method to check for changes in set value being passed in
    check_if_changed = True

    for param_key, param_value in i.items():
        # print(f'param_key is {param_key} and param_value is {param_value}')

        # print(f'temp is {i["Temp"]}')

        # create the class name from param_key
        # curr_param_class = globals()[param_key]
        curr_param_class = eval('prm_cls' + '.' + param_key)

        # create class instance from param_key
        curr_param_class_instance = curr_param_class(name=param_key)

        is_value_set = curr_param_class_instance.set_param(param_key, param_value, check_if_changed)

        if is_value_set:
            check_if_changed = False

    print('All params are set. Ready to collect data!!!')

    datalogger.log_data(swp_gbl.curr_params, swp_gbl.shr_logs)

print('Sweep is complete.')
