"""
@author: ksanoo
@updated_at: 12/4/2020
@description: All parameters that are to be swept (specified in sweep_setup files) must have a class definition in
this script. The class name must be the same as the parameter key in loop_param
"""

import global_vars as swp_gbl
from parameter_classes import GenericParam
import device_startup
from abc import abstractmethod


class TestParameter12(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().check_value_change(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            print('>>> Executing TestParameter1.set_param...')
            pass

        return value_changed


class RxParameter1(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().check_value_change(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            pass

        return value_changed
