import os
from collections import OrderedDict

dut_name = 'MxFE_R2B_SD'
board_name = 'RevJ_SD'
JIRA_task_no = os.path.basename(__file__)[5:9]
misc_tag = 'Fs_Fin_Sweep'

# add more loop params here
# the loop param objects created must have the same name as the key entered here
loop_param = OrderedDict()

loop_param['Temp'] = [-10]
loop_param['Fadc'] = [4000, 6000]
loop_param['StartupCount'] = [1, 2]
loop_param['Fin'] = [4500, 2234]
