import numpy as np
from basic_device import RFGeneratorStats, UnitFreq
import labscript_utils.h5_lock
import labscript_utils.properties
import h5py

from blacs.tab_base_classes import Worker
import labscript_utils.properties
from .agilent_4422B_device import AgilentE4422BDevice


class AgilentE4422BWorker(GPIBWorker):

    def init(self):
        # TODO Add a print GPIB TO CHECK ORDER
        #  print( " THE WORKER EXPERIMENt : THE GPIB SUPER WORKER WAS CALLED ")  
        self.device     = AgilentE4422BDevice( self.GPIB_connection.write , self.GPIB_connection.query)
        print( " THE WORKER EXPERIMENt : AGILENT E4422B WAS CALLED ") 

    # -------------------------- The usuals
    def program_manual(self, front_panel_values):
        frequency_hz = float(front_panel_values["rf/frequency"])
        power_dbm    = float(front_panel_values["rf/power"])

        self.device.set_freq(frequency_hz)
        self.device.set_power(power_dbm)

        return {"rf/frequency": frequency_hz, "rf/power": power_dbm }



    def transition_to_buffered(self, device_name, h5_filepath, initial_values, fresh):

        # For remote worker to find correct path:
        if getattr(self, "is_remote", False):
            h5_filepath = path_to_local(h5_filepath)

        # Start from manual-mode values, then overwrite with shot values from HDF5
        # TODO Check this dance beetween inital values shot values and outputtable looks weird
        dtypes = [ ("frequency_hz", np.float64),
                   ("power_dbm", np.float32),
        ]

        output_table = np.zeros(1, dtype=dtypes)
        output_table["frequency_hz"] = initial_values["rf/frequency"]
        output_table["power_dbm"] = initial_values["rf/power"]

        # Get values from experiment script / shot file
        with h5py.File(h5_filepath, "r") as hdf5_file:
            group = hdf5_file["devices"][device_name]
            shot_values = group["OUTPUT_DATA"][0]

            output_table["frequency_hz"] = shot_values["frequency_hz"]
            output_table["power_dbm"] = shot_values["power_dbm"]

        frequency_hz = float(output_table["frequency_hz"][0])
        power_dbm = float(output_table["power_dbm"][0])

        # Send values to instrument TODO 
        # OLD self.send_GPIB_voltage(voltage=output_table['v%d' % (i + 1)], output=i + 1)
        self.device.set_freq(frequency_hz)
        self.device.set_power(power_dbm)

        # Return final values for transition_to_manual()
        self.final_values = {   "rf/frequency": frequency_hz,
                                "rf/power": power_dbm }

        return self.final_values


    def transition_to_manual(self, abort=False):
        values = getattr(self, "final_values", None)

        if values is None:
            return True

        frequency_hz = float(values["rf/frequency"])
        power_dbm = float(values["rf/power"])

        self.device.set_freq(frequency_hz)
        self.device.set_power(power_dbm)

        return True
    

    # ------------------------------------------ Blacs Tabs functions
    
    def apply_to_device(self, stats : RFGeneratorStats):
        print(stats, print(type(stats)))
        # TODO 
        # self.device.set_freq(stats.freq_mhz , UnitFreq.MHZ)
        # self.device.set_power(stats.power_dbm)
        

    def read_stats_from_device(self) -> RFGeneratorStats:
        freq_read = self.device.get_freq()
        freq_mhz = freq_read / 1e6
        power_dbm = self.device.get_power()
        rf_on = self.device.get_output_rf()
        return RFGeneratorStats(freq_mhz=freq_mhz , power_dbm=power_dbm , rf_on=rf_on)


    def set_output_rf(self, state):
        print(state , type(state))
        # TODO  
        # self.device.set_output_rf(state)





