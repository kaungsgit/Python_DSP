"""
@author: ksanoo
@updated_at: 12/4/2020
@description: All parameters that are to be swept (specified in sweep_setup files) must have a class definition in
this script. The class name must be the same as the parameter key in loop_param
"""

import global_vars as swp_gbl
import device_startup
from abc import abstractmethod


def bf_write(bf_name, value):
    print('Executing bf_write({}, {})...'.format(bf_name, value))


class GenericParam:
    """
    Abstract class that contains common code and an abstract method set_param, which the derived classes will implement
    """

    def __init__(self, name='generic_param'):
        self.key = name
        self.value = None

    def check_value_change(self, key=None, value=None, check_if_value_changed=True):
        """
        Checks if the value to be set has changed from prev iteration by comparing with the value in curr_params dict
        (this is to avoid resetting a parameter's value if it hasn't changed yet).
        Also updates curr_params dict accordingly.

        Args:
            key (str): name of parameter
            value (bool,int,float): value to be set for the parameter
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise
        """

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

    @abstractmethod
    def set_param(self, key=None, value=None, check_if_value_changed=True):
        """
        Abstractmethod for implementing what needs to happen when a sweep parameter is set.

        Args:
            key (str): name of parameter, generally the same as derived param class name (except for GenericBitfieldWrite class)
            value (bool,int,float): value to be set for the parameter
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            None:

        Raises:
            NotImplementedError
        """

        raise NotImplementedError


class TestParameter1(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().check_value_change(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            print('>>> Executing TestParameter1.set_param...')
            pass

        return value_changed


class TestParameter2(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        value_changed = super().check_value_change(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            print('>>> Executing TestParameter2.set_param...')
            pass

        return value_changed


class GenericBitfieldWrite(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        """
        Checks if the value to be set has changed from prev iteration.
        Sets the key (bit field name) to the value passed in if needed.
        Every bit field sweep is handled by this class. This is the only class so far that uses the key argument to perform set_param.

        Args:
            key (str): bit field name
            value (float): bit field value in hexadecimal
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise

        """
        value_changed = super().check_value_change(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            bf_write(key, value)  # bf_ has been removed from key
            pass

        return value_changed


class Temp(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        """
        Checks if the value to be set has changed from prev iteration.
        Sets the Temp parameter to the value passed in if needed.

        Args:
            key (str): Temp
            value (float): temperature value in degC
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise

        """
        value_changed = super().check_value_change(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # oven.set_temp(value) or sensata.set_temp(value)
            if value == -40:
                swp_gbl.shr_data['some_temp_related_var'] = 1024
            else:
                swp_gbl.shr_data['some_temp_related_var'] = 512

        return value_changed


class Fadc(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        """
        Checks if the value to be set has changed from prev iteration.
        Sets the Fadc parameter to the value passed in if needed.

        Args:
            key (str): Fadc
            value (float): ADC sampling rate in MHz
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise
        """
        value_changed = super().check_value_change(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # DUT.set_fadc(value)

            if swp_gbl.shr_data['some_temp_related_var'] == 1024:
                print('Brace the DUT. Winter is coming!')
            else:
                pass

        return value_changed


class StartupCount(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        """
        Checks if the value to be set has changed from prev iteration.
        Sets the StartupCount parameter to the value passed in if needed. Each time this param is set, the DUT startup
        script is called.

        Args:
            key (str): StartupCount
            value (int): index of startup count (1=first startup, 2=second startup, etc...)
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise
        """
        value_changed = super().check_value_change(key, value, check_if_value_changed)

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
        """
        Checks if the value to be set has changed from prev iteration.
        Sets the Fin parameter to the value passed in if needed.

        Args:
            key (str): Fin
            value (float): analog input frequency in MHz
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise
        """
        value_changed = super().check_value_change(key, value, check_if_value_changed)

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
        """
        Checks if the value to be set has changed from prev iteration.
        Sets the Ain parameter to the value passed in if needed.

        Args:
            key (str): Ain
            value (float): analog output amplitude in dBFS (amp servo performed in this function)
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise
        """
        value_changed = super().check_value_change(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # testbench.sigGen.set_amplitude(value) or
            # kernel.servo_amplitude(value, tolerance=0.1)
            pass

        return value_changed


class ADCCalibrationState(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        """
        Checks if the value to be set has changed from prev iteration.
        Sets the ADCCalibrationState parameter to the value passed in if needed.

        Args:
            key (str): ADCCalibrationState
            value (bool): enable/disable ADC cal
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise
        """
        value_changed = super().check_value_change(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # TxFE.configure_calibration(value)
            pass
        return value_changed


class SupplyPct(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        """
        Checks if the value to be set has changed from prev iteration.
        Sets the SupplyPct parameter to the value passed in if needed.

        Args:
            key (str): SupplyPct
            value (float): percentage in supply voltage change (0 -> nominal supply, -5 -> -5% lower than nominal)
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise
        """
        value_changed = super().check_value_change(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # testbench.set_supplies(value)

            # example logging using shr_datalogs
            swp_gbl.shr_datalogs['AVDD1 Voltage'] = 1 + (value * 0.01)
            pass

        return value_changed


class DACState(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        """
        Checks if the value to be set has changed from prev iteration.
        Sets the DACState parameter to the value passed in if needed.

        Args:
            key (str): DACState
            value (bool): enable/disable DAC
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise
        """
        value_changed = super().check_value_change(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            pass
        return value_changed


class DACFout(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        """
        Checks if the value to be set has changed from prev iteration.
        Sets the DACFout parameter to the value passed in if needed.

        Args:
            key (str): DACFout
            value (float): DAC output tone frequency in MHz
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise
        """
        value_changed = super().check_value_change(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            pass
        return value_changed


class DACPwr(GenericParam):

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        """
        Checks if the value to be set has changed from prev iteration.
        Sets the DACPwr parameter to the value passed in if needed.

        Args:
            key (str): DACPwr
            value (float): DAC output tone amplitude in dBm
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise
        """
        value_changed = super().check_value_change(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            pass
        return value_changed


class Fin1(GenericParam):
    """ Everything is the same as Fin except you're referencing a different sig generator """

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        """
        Checks if the value to be set has changed from prev iteration.
        Sets the Fin1 parameter to the value passed in if needed.

        Args:
            key (str): Fin1
            value (float): analog input (secondary) frequency in MHz
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise
        """
        value_changed = super().check_value_change(key, value, check_if_value_changed)

        if value_changed:
            # custom set_param function starts here
            # testbench.sigGen1.set_frequency(value)

            pass
        return value_changed


class Ain1(GenericParam):
    """ Everything is the same as Ain except you're referencing a different sig generator """

    def set_param(self, key=None, value=None, check_if_value_changed=True):
        """
        Checks if the value to be set has changed from prev iteration.
        Sets the Ain1 parameter to the value passed in if needed.

        Args:
            key (str): Ain1
            value (float): analog output (secondary) amplitude in dBFS (amp servo performed in this function)
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise
        """
        value_changed = super().check_value_change(key, value, check_if_value_changed)
        if value_changed:
            # custom set_param function starts here
            # testbench.sigGen1.set_amplitude(value) or
            # kernel.servo_amplitude(value, tolerance=0.1)
            pass

        return value_changed


class Ain_Ain1(GenericParam):
    def set_param(self, key=None, value=None, check_if_value_changed=True):
        """
        Checks if the value to be set has changed from prev iteration.
        Pair value setting. Bind Ain and Ain1 parameters.
        Sets value[0] to Ain and value[1] to Ain1 if needed.

        Args:
            key (str): Ain_Ain1
            value (list): analog output power pair in dBFS
            check_if_value_changed (bool): checks if the value to be set has changed from prev iteration

        Returns:
            True if value change is detected. False otherwise
        """
        value_changed = super().check_value_change(key, value, check_if_value_changed)

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
