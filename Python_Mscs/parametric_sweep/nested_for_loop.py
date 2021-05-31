"""
@author: ksanoo
@updated_at: 5/29/2021
@description: example code for showing traditional for nested loop sweeping (compare this with my parametric sweep)
"""

temps = [125, -40]
Fadc_list = [6000, 8000]


def set_temp(temp_val):
    print(f'Setting Temp to {temp_val}...')


def set_sample_rate(sample_rate_val):
    print(f'Setting Fadc to {sample_rate_val}...')


def collect_data():
    print('************ All params are set. Ready to collect data!!!*****************')


for temp in temps:
    set_temp(temp)
    if temp == -40:
        some_temp_related_var = 1024
    else:
        some_temp_related_var = 512

    for Fadc in Fadc_list:
        set_sample_rate(Fadc)
        if some_temp_related_var == 1024:
            print('Brace the DUT. Winter is coming!')
            pass

        collect_data()

print('Sweep is complete.')
