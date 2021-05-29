"""
@author: ksanoo
@updated_at: 12/4/2020
@description: All parameters that are to be swept (specified in sweep_setup files) must have a class definition in
this script. The class name must be the same as the parameter key in loop_param
"""

import global_vars as swp_gbl
import device_startup


def bf_write(bf_name, value):
    print('Executing bf_write({}, {})...'.format(bf_name, value))


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


class TestParameter1(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            print('>>> Executing TestParameter1.set_param...')
            pass

        return value_changed


class TestParameter2(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            print('>>> Executing TestParameter2.set_param...')
            pass

        return value_changed


class BitfieldWrite(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            bf_write(key, value)  # bf_ has been removed from key
            pass

        return value_changed


class Temp(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # oven.set_temp(value)
            # sensata.set_temp(value)
            if value == -40:
                swp_gbl.shr_data['some_temp_related_var'] = 1024
            else:
                swp_gbl.shr_data['some_temp_related_var'] = 512

        return value_changed


class Fadc(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            pass

            if swp_gbl.shr_data['some_temp_related_var'] == 1024:
                print('Brace TxFEs. Winter is coming!')
            else:
                pass

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

            device_startup.startup_dut(swp_gbl.curr_params['Fadc'], swp_gbl.curr_params['DACState'])
        return value_changed


class Fin(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # testbench.sigGen.set_frequency(value)

            # if swp_gbl.shr_data['val5'] == 1010:
            #     # print('doing some extra stuff in setting Fin')
            #     swp_gbl.shr_datalogs['Fin_readback'] = 1
            # else:
            #     swp_gbl.shr_datalogs['Fin_readback'] = 0
            pass
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

            # example logging using shr_datalogs
            swp_gbl.shr_datalogs['AVDD1 Voltage'] = 1 + (value * 0.01)
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


class DACPwr(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            pass
        return value_changed


class Fin1(GenericParam):
    """ Everything is the same as Fin except you're referencing a different sig generator """

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # testbench.sigGen1.set_frequency(value)

            pass
        return value_changed


class Ain1(GenericParam):
    """ Everything is the same as Ain except you're referencing a different sig generator """

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # testbench.sigGen1.set_amplitude(value) or
            # kernel.servo_amplitude(value, tolerance=0.1)
            pass

        return value_changed


class Ain_Ain1(GenericParam):
    """ Bind Ain and Ain1. Value is now an array of two elements """

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().set_param(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # testbench.sigGen1.set_amplitude(value) or
            # kernel.servo_amplitude(value, tolerance=0.1)

            Ain_inst = Ain()
            Ain1_inst = Ain1()

            Ain_inst.set_param(value[0])
            Ain1_inst.set_param(value[1])

            pass

        return value_changed
