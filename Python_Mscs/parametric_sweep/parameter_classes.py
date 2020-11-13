import globals as swp_gbl
import device_startup


class GenericParams:
    shr_data = dict()
    curr_params = dict()

    def __init__(self, name='generic_param'):
        self.key = name
        self.value = -1000

    def set_param(self, key=None, value=None, check_if_changed=True):

        if check_if_changed:

            param_changed = True

            if key in swp_gbl.curr_params.keys():
                if swp_gbl.curr_params[key] == value:
                    param_changed = False
                    # print('{} was already set to {}...'.format(key, value))
                else:
                    # prev value of key is not the same as the current value
                    print('Setting {} to {}...'.format(key, value))
                    self.value = value
                    swp_gbl.curr_params[key] = value
            else:
                # key is not in dict yet, create key and assign value
                print('Setting {} to {}...'.format(key, value))
                self.value = value
                swp_gbl.curr_params[key] = value
        else:
            # don't check, just set it
            param_changed = True
            print('Force setting {} to {}...'.format(key, value))
            self.value = value
            swp_gbl.curr_params[key] = value

        return param_changed


class Temp(GenericParams):

    def set_param(self, key=None, value=None, check_if_changed=True):
        param_changed = super().set_param(key, value, check_if_changed)

        if param_changed:
            if value == -10:
                # set some class attribute that other child class will use in its set_param method
                swp_gbl.shr_logs['val1'] = 10
                swp_gbl.shr_logs['val5'] = 1010
                # print('Class attri 2 is changed in setting Temp')
            else:
                GenericParams.shr_data['val1'] = -1

        return param_changed


class Fadc(GenericParams):

    def set_param(self, key=None, value=None, check_if_changed=True):
        param_changed = super().set_param(key, value, check_if_changed)

        if param_changed:
            if swp_gbl.shr_logs['val1'] == 10:
                # print('doing some extra stuff in setting Fadc')
                pass

        return param_changed


class StartupCount(GenericParams):

    def set_param(self, key=None, value=None, check_if_changed=True):
        param_changed = super().set_param(key, value, check_if_changed)

        if param_changed:
            device_startup.startup_dut()
        return param_changed


class Fin(GenericParams):

    def set_param(self, key=None, value=None, check_if_changed=True):
        param_changed = super().set_param(key, value, check_if_changed)

        if param_changed:
            if swp_gbl.shr_logs['val5'] == 1010:
                # print('doing some extra stuff in setting Fin')
                pass

        return param_changed
