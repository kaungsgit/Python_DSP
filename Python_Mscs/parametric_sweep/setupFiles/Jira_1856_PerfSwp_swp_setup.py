import os
from collections import OrderedDict

swp_info = OrderedDict()

swp_info['dut_name'] = 'TxFE_R2B_SD'
swp_info['board_name'] = 'RevJ_SD'
swp_info['JIRA_task_no'] = os.path.basename(__file__)[5:9]
swp_info['JIRA_task_descr'] = 'TxFE_perf_swp'
swp_info['misc_tag'] = 'Fs_Fin_Sweep'

# add more loop params here
# the loop param objects created must have the same name as the key entered here
loop_param = OrderedDict()
loop_param['Temp'] = [125, -40]
loop_param['Fadc'] = [4000, 6000]
loop_param['DACState'] = [False]
loop_param['DACFout'] = [None]
loop_param['StartupCount'] = [1, 2]
loop_param['ADCCalibrationState'] = [True, False]
loop_param['Fin'] = [4500, 2234]
loop_param['Ain'] = [-1]
