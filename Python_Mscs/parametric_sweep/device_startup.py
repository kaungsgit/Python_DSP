# Device startup script
def startup_dut(Fadc, DACState, DACFout):
    # TxFE.configure_ADC(Fadc)
    # if DACState is True:
    #     TxFE.configure_DAC(DACFout)
    print('Device startup is complete with Fadc {}.'.format(Fadc))


if __name__ == '__main__':
    startup_dut()
