from .agilent_4422B_device import AgilentE4422BDevice
from basic_device import UnitFreq


''' Stand alone Testing Script for the Device'''

if __name__ == "__main__":


    addr = ""
    dev = AgilentE4422BDevice(write = None ,query = None , addr = addr)

    # --- Testing Zone

    # dev.set_freq(300 , UnitFreq.KHZ)

    # dev.modulate_internal_freq(
    #     carrier         = 300,
    #     carrier_unit    = UnitFreq.KHZ,
    #     power_dbm       = -18         , 
    #     deviation       = 10
    # )

    dev.sweep_step_frequency_config(
        start   = 260 ,
        stop    = 340 , 
        points  = 9   ,
        dwell_s = 1,
        unit    = UnitFreq.KHZ,
        power_dbm = -20,
        continuous = False
    )

    dev.sweep_start_immediate()


    # --- THE END
    dev.rm.close()
    dev.GPIB_connection.close()


    

