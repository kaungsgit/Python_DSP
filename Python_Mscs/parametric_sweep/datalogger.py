import csv
from collections import OrderedDict
import numpy as np
import os

csv_file = "mycsvfile.csv"


def merge_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


def log_data(curr_params, shr_logs):
    analysis_collection = OrderedDict()

    analysis_collection_1 = merge_dicts(curr_params, shr_logs)

    analysis_collection_1['SNR'] = 0.2 * np.random.randn() + 56

    exists = os.path.exists(csv_file)

    if exists:
        with open(csv_file, 'a', newline="") as f:
            w = csv.DictWriter(f, analysis_collection_1.keys())
            w.writerow(analysis_collection_1)

    else:
        with open(csv_file, 'w', newline="") as f:
            w = csv.DictWriter(f, analysis_collection_1.keys())
            w.writeheader()
            w.writerow(analysis_collection_1)
