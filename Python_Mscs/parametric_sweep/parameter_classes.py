"""
@author: ksanoo
@updated_at: 12/4/2020
@description: All parameters that are to be swept (specified in sweep_setup files) must have a class definition in
this script. The class name must be the same as the parameter key in loop_param
"""

import global_vars as swp_gbl
import device_startup


class GenericParam:

    def __init__(self, name='generic_param'):
        self.key = name
        self.value = -1000

    def set_param(self, key=None, value=None, check_if_value_changed=True):

        if check_if_value_changed:

            if key in swp_gbl.curr_params.keys():
                if swp_gbl.curr_params[key] == value:
                    value_changed = False
                    # print('{} was already set to {}...'.format(key, value))
                else:
                    # prev value of key is not the same as the current value
                    print('Setting {} to {}...'.format(key, value))
                    self.value = value
                    swp_gbl.curr_params[key] = value
                    value_changed = True
            else:
                # key is not in dict yet, create key and assign value
                print('Setting {} to {}...'.format(key, value))
                self.value = value
                swp_gbl.curr_params[key] = value
                value_changed = True
        else:
            # don't check, just set it
            value_changed = True
            print('Force setting {} to {}...'.format(key, value))
            self.value = value
            swp_gbl.curr_params[key] = value

        return value_changed


class Temp(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # oven.set_temp(value)
            if value == -10:
                # set some variable that other param classes will use
                swp_gbl.shr_data['val1'] = 10
                swp_gbl.shr_data['val5'] = 1010
            else:
                swp_gbl.shr_data['val1'] = -1
                swp_gbl.shr_data['val5'] = -5

        return value_changed


class Fadc(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            if swp_gbl.shr_data['val1'] == 10:
                # print('doing some extra stuff in setting Fadc')
                swp_gbl.shr_datalogs['Fadc_readback'] = 1
            else:
                swp_gbl.shr_datalogs['Fadc_readback'] = 0

        return value_changed


class StartupCount(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here

            # check if Fadc has been defined prior StartupCount or not
            if 'Fadc' in swp_gbl.curr_params:
                print('Fadc is defined before StartupCount.')
            else:
                print('Fadc is not defined before StartupCount.')
                default_Fadc = 6000
                swp_gbl.curr_params['Fadc'] = default_Fadc
                print(
                    'Assuming Fadc is to be changed after startup, Fadc has been defined to be default value of {}'.format(
                        default_Fadc))

            device_startup.startup_dut(swp_gbl.curr_params['Fadc'], swp_gbl.curr_params['DACState'],
                                       swp_gbl.curr_params['DACFout'])
        return value_changed


class Fin(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # testbench.sigGen.set_frequency(value)
            if swp_gbl.shr_data['val5'] == 1010:
                # print('doing some extra stuff in setting Fin')
                swp_gbl.shr_datalogs['Fin_readback'] = 1
            else:
                swp_gbl.shr_datalogs['Fin_readback'] = 0

        return value_changed


class Ain(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # testbench.sigGen.set_amplitude(value) or
            # kernel.servo_amplitude(value, tolerance=0.1)
            pass

        return value_changed


class ADCCalibrationState(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # TxFE.configure_calibration(value)
            pass
        return value_changed


class SupplyPct(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # testbench.set_supplies(value)
            pass
        return value_changed


class DACState(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            pass
        return value_changed


class DACFout(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            pass
        return value_changed
