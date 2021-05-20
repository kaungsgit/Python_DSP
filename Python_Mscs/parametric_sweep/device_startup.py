"""
@author: ksanoo
@updated_at: 12/4/2020
@description: Golden DUT startup script
"""


def startup_dut(Fadc, DACState):
    # sig_gen.set_freq(Fadc)
    # TxFE.configure_ADC(Fadc)
    # if DACState is True:
    #     TxFE.configure_DAC(DACFout)
    print('Device startup is complete with Fadc {}.'.format(Fadc))


if __name__ == '__main__':
    Fadc = 6e3
    DACState = False
    DACFout = None
    startup_dut(Fadc, DACState)
