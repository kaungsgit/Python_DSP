import os
from collections import OrderedDict
import global_vars as swp_gbl

swp_info = OrderedDict()

swp_info['dut_name'] = 'TxFE_R2B_SD'
swp_info['board_name'] = 'RevF_SD'
swp_info['JIRA_task_no'] = os.path.basename(__file__)[5:9]
swp_info['JIRA_task_descr'] = 'TxFE_perf_swp'
swp_info['misc_tag'] = 'Fs_Fin_Sweep'

# add more loop params here
# the loop param objects created must have the same name as the key entered here
loop_param = OrderedDict()

loop_param['Temp'] = [125, -40]
loop_param['Fadc'] = [6000, 8000]

loop_param['DACState'] = [False, True]
loop_param['StartupCount'] = [1, 2]

loop_param['SupplyPct'] = [-5]
loop_param['ADCCalibrationState'] = [True]
loop_param['DACFout'] = [1027]
loop_param['DACPwr'] = [-9]

loop_param['Fin'] = [263]
loop_param['Ain'] = [-1, -6]
loop_param['Fin1'] = [363]
loop_param['Ain1'] = [-1, -6]

# loop_param['Sweep_bitField_A'] = [0, 1, 2, 3]
# loop_param['Sweep_bitField_B'] = [4, 5, 6, 7]


# includes some variables that can be specific to a sweep
swp_gbl.shr_data['dummy data'] = 123
