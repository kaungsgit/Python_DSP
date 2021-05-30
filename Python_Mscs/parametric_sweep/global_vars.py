"""
@author: ksanoo
@updated_at: 12/4/2020
@description: global variables for generic sweep.
shr_data contains shared data between parameter classes.
shr_datalogs contains conditional datalogs for certain parameters.
curr_params contains the current parameters being set.
"""

from collections import OrderedDict
import os

datalog_path = os.path.join(os.getcwd(), 'datalogs')

shr_data = OrderedDict()
shr_datalogs = OrderedDict()
curr_params = OrderedDict()
