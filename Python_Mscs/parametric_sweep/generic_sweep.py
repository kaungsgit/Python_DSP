import numpy as np
from collections import OrderedDict

# to compute all possible permutations
# using itertools.product()
import itertools

import global_vars
import parameter_classes as prm_cls
import datalogger
import os
import importlib

swpSetupFiles = ['Jira_1856_PerfSwp_swp_setup', 'Jira_1857_changeFadcFast_swp_setup', 'Jira_1858_noFin_swp_setup',
                 'Jira_1860_VT_FullChip_swp_setup', 'Jira_1859_VTFs_swp_setup']
# swpSetupFiles = ['Jira_1856_PerfSwp_swp_setup']

for file in swpSetupFiles:
    swpSetup = importlib.import_module('setupFiles.' + file)

    swp_gbl = importlib.reload(global_vars)
    datalog_path = swp_gbl.datalog_path
    loop_param = swpSetup.loop_param
    swp_info = swpSetup.swp_info
    make_dir = True

    keys, values = zip(*loop_param.items())
    permutations = [OrderedDict(zip(keys, v)) for v in itertools.product(*values)]

    # create result folders and files
    out_file = datalogger.OutputFile(swp_info['dut_name'], swp_info['board_name'], swp_info['misc_tag'],
                                     swp_info['JIRA_task_no'], swp_info['JIRA_task_descr'], datalog_path)
    sweep_file_name = out_file.create_name(sweep_vars='', is_plot=False)
    sweep_file_path, fft_pic_path, fft_data_path = out_file.create_paths(mk_dir=make_dir)
    results_file_path = os.path.join(sweep_file_path, sweep_file_name)

    for curr_perm in permutations:
        # this makes the set_param method to check for changes in set value being passed in
        print(curr_perm)
        check_if_value_changed = True

        for param_key, param_value in curr_perm.items():
            # print(f'param_key is {param_key} and param_value is {param_value}')

            # create the class name from param_key
            curr_param_class = eval('prm_cls' + '.' + param_key)
            # create class instance from param_key
            curr_param_class_instance = curr_param_class(name=param_key)
            # call set_param method on each parameter class
            value_changed = curr_param_class_instance.set_param(param_key, param_value, check_if_value_changed)

            if value_changed:
                check_if_value_changed = False

        print('************ All params are set. Ready to collect data!!!*****************')

        datalogger.log_data(results_file_path, swp_info, swp_gbl.curr_params, swp_gbl.shr_datalogs)

print('Sweep is complete.')
