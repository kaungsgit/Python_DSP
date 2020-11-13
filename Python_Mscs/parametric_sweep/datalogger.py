import csv
from collections import OrderedDict
import numpy as np
import os
from datetime import datetime
import globals as swp_gbl


class OutputFile:

    def __init__(self, dut_name, board_name, misc_tag, JIRA_task_no, JIRA_task_descr, base_path):
        self.dut_name = dut_name
        self.board_name = board_name
        self.misc_tag = misc_tag
        self.JIRA_task_no = JIRA_task_no
        self.base_path = base_path
        self.JIRA_task_name = 'JIRA_' + str(JIRA_task_no)
        self.JIRA_task_descr = JIRA_task_descr
        self.curr_date = datetime.now().replace(microsecond=0)
        self.curr_task_path = os.path.join(self.base_path, str(self.JIRA_task_name) + '_' + str(self.JIRA_task_descr),
                                           str(self.curr_date).replace(':', '') + '_{}'.format(self.misc_tag))
        self.csv_data_path = None
        self.fft_pic_path = None
        self.fft_data_path = None

    def create_name(self, sweep_vars, is_plot):
        if is_plot:
            file_name = self.dut_name + '-' + self.board_name + '-' + sweep_vars + '-' + self.misc_tag + '-' + self.JIRA_task_name + '.png'
        else:
            file_name = self.dut_name + '-' + self.board_name + '-' + sweep_vars + '-' + self.misc_tag + '-' + self.JIRA_task_name + '.csv'
        return file_name

    def create_paths(self, mk_dir=True):

        self.csv_data_path = os.path.join(self.curr_task_path, 'SwpData')
        self.fft_pic_path = os.path.join(self.curr_task_path, 'FFTPics')
        self.fft_data_path = os.path.join(self.curr_task_path, 'FFTRD')

        if mk_dir:
            try:
                os.makedirs(self.csv_data_path)
                os.makedirs(self.fft_pic_path)
                os.makedirs(self.fft_data_path)
            except OSError:
                if not (os.path.isdir(self.csv_data_path) and
                        os.path.isdir(self.csv_data_path) and
                        os.path.isdir(self.csv_data_path)):
                    raise Exception('Could not create file path')

        return self.csv_data_path, self.fft_pic_path, self.fft_data_path


def merge_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


def log_data(results_file_path, swp_info, curr_params, shr_logs):
    analysis_collection = merge_dicts(swp_info, curr_params)
    analysis_collection = merge_dicts(analysis_collection, shr_logs)

    # Dummy DUT measurement

    if swp_gbl.curr_params['CalibrationUsed'] is True:
        analysis_collection['SNR'] = 0.2 * np.random.randn() + 56
    else:
        analysis_collection['SNR'] = 0.2 * np.random.randn() + 54

    exists = os.path.exists(results_file_path)

    if exists:
        with open(results_file_path, 'a', newline="") as f:
            w = csv.DictWriter(f, analysis_collection.keys())
            w.writerow(analysis_collection)

    else:
        with open(results_file_path, 'w', newline="") as f:
            w = csv.DictWriter(f, analysis_collection.keys())
            w.writeheader()
            w.writerow(analysis_collection)
