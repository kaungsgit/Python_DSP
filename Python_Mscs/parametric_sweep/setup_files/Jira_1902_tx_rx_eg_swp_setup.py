import os
from collections import OrderedDict
import global_vars as swp_gbl

swp_info = OrderedDict()

swp_info['dut_name'] = 'TxFE_R2B_SD'
swp_info['board_name'] = 'RevF_SD'
swp_info['JIRA_task_no'] = os.path.basename(__file__)[5:9]
swp_info['JIRA_task_descr'] = 'tx_rx_sweep'
swp_info['misc_tag'] = 'tx_rx_sweep'

# add more loop params here
# the loop param objects created must have the same name as the key entered here
loop_param = OrderedDict()
loop_param['Temp'] = [-25, 125]
loop_param['bf_pll_en'] = [0, 1]
# loop_param['bf_rx_clk_sel'] = [0, 1, 2, 3]
loop_param['TxParameter1'] = [-4, 4]
loop_param['RxParameter1'] = [-2, 2]
