"""
@author: ksanoo
@updated_at: 12/4/2020
@description: Main script for running the parametric sweep code.
Sweep setup files, where what parameter and what values to be swept are defined, are located under setup_files.
Parameter classes with set_param methods are in parameter_classes.py.
.csv data files will appear under the folder datalogs.
"""

import numpy as np
from collections import OrderedDict

# to compute all possible permutations
# using itertools.product()
import itertools

import global_vars
import parameter_classes, parameter_classes_rx, parameter_classes_tx
import datalogger
import os
import importlib
import sys
import inspect


def search_avail_param_classes():
    """
    Search all available parameter classes from modules that start with 'parameter_classes' and returns a dict
    that contains all available parameters.

    Returns: Dict that contains all available parameters.

    Raises: Exception if duplicate parameter class definitions are found within parameter_classes modules.

    """
    param_modules = dict(filter(lambda item: 'parameter_classes' in item[0], sys.modules.items()))

    avail_param_classes = {}
    for param_module in param_modules:
        param_classes_per_module = dict(inspect.getmembers(sys.modules[param_module], inspect.isclass))
        param_classes_per_module_set = set(param_classes_per_module)
        avail_param_classes_set = set(avail_param_classes)

        for name in param_classes_per_module_set.intersection(avail_param_classes_set):
            if name is not 'GenericParam':
                raise Exception(
                    f'Duplicate param class {param_classes_per_module[name]} found! \nRename or delete {name} class.')

        avail_param_classes.update(param_classes_per_module)

    return avail_param_classes


# dict containing all available param classes
avail_param_classes = search_avail_param_classes()

# swp_setup_files = ['Jira_1856_PerfSwp_swp_setup', 'Jira_1857_changeFadcFast_swp_setup', 'Jira_1858_noFin_swp_setup',
#                  'Jira_1860_VT_FullChip_swp_setup', 'Jira_1859_VTFs_swp_setup']
# swp_setup_files = ['Jira_1856_PerfSwp_swp_setup']

# swp_setup_files = ['Jira_1900_forloop_vs_myframework_swp_setup']
# swp_setup_files = ['Jira_1901_bitfieldWrite_example_swp_setup']
swp_setup_files = ['Jira_1902_tx_rx_eg_swp_setup']

# run multiple sweep setup files
for file in swp_setup_files:

    # reload global_vars to re-initialize them to blanks after each sweep setup file
    swp_gbl = importlib.reload(global_vars)

    swpSetup = importlib.import_module('setup_files.' + file)

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
        # print(curr_perm)
        check_if_value_changed = True

        for param_key, param_value in curr_perm.items():
            # print(f'param_key is {param_key} and param_value is {param_value}')

            if param_key[0:3] == 'bf_':
                # for all bitfield writes, the class GenericBitfieldWrite is used
                curr_param_class = avail_param_classes['GenericBitfieldWrite']
                param_key = param_key[3:]

            else:
                # create the class from param_key
                curr_param_class = avail_param_classes[param_key]

            # create class instance from param_key
            curr_param_class_instance = curr_param_class(name=param_key)
            # call set_param method on each parameter class
            value_changed = curr_param_class_instance.set_param(param_key, param_value, check_if_value_changed)

            if value_changed:
                # once outer loop value has changed, following inner loops must execute,
                # so don't bother checking for value change
                check_if_value_changed = False

        print('************ All params are set. Ready to collect data!!!*****************')

        datalogger.log_data(results_file_path, swp_info, swp_gbl.curr_params, swp_gbl.shr_datalogs)

print('Sweep is complete.')
