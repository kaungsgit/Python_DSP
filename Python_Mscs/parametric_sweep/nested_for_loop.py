temps = [125, -40]
sample_rates = [6000, 8000]


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

    for sample_rate in sample_rates:
        set_sample_rate(sample_rate)
        if some_temp_related_var == 1024:
            print('Brace TxFEs. Winter is coming!')
            pass

        collect_data()

print('Sweep is complete.')
