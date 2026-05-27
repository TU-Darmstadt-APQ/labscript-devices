import numpy as np
# from labscript_utils.shared_drive import path_to_local  # TODO check import
import labscript_utils.h5_lock
import labscript_utils.properties
import h5py

from blacs.tab_base_classes import Worker
import labscript_utils.properties

from .agilent_4422B_device import AgilentE4422BDevice
from .basic_device import RFGeneratorStats, UnitFreq
from ..GPIBDevice import GPIBWorker, EosStrategy


class AgilentE4422BWorker(GPIBWorker):

    def init(self):
        super().init()
        self.device             = AgilentE4422BDevice( self.GPIB_connection.write , self.GPIB_connection.query)
        self.initial_state      = self.read_stats_from_device()


    # -------------------------- The usuals
    def program_manual(self, front_panel_values):
        frequency_hz = float(front_panel_values["rf/frequency"])
        power_dbm    = float(front_panel_values["rf/power"])

        self.device.set_freq(frequency_hz)
        self.device.set_power(power_dbm)

        return {"rf/frequency": frequency_hz, "rf/power": power_dbm }


    def program_manual(self, front_panel_values):
        frequency_hz = float(front_panel_values["rf_freq"])
        power_dbm = float(front_panel_values["rf_amp"])


        print(f" LabWork Frequency {frequency_hz}")
        print(f" LabWork amp {power_dbm}")

        # TODO CHECK VALUES FIRST
        # self.device.set_freq(frequency_hz)
        # self.device.set_power(power_dbm)


        return { "rf_freq": frequency_hz,"rf_amp": power_dbm}


    def transition_to_buffered(self, device_name, h5_filepath, initial_values, fresh):
        print(" ===== TRANSITION TO BUFFERED ")
        if getattr(self, "is_remote", False):
            h5_filepath = path_to_local(h5_filepath)

        with h5py.File(h5_filepath, "r") as hdf5_file:

            if device_name not in hdf5_file["devices"]:
                print(f"{device_name}: not in shot file, leaving unchanged.")
                return initial_values

            group = hdf5_file["devices"][device_name]
            shot_values = group["OUTPUT_DATA"][0]

            frequency_hz = float(shot_values["frequency_hz"])
            power_dbm = float(shot_values["power_dbm"])

        # TODO CHECK THESE VALUES 
        # self.device.set_freq(frequency_hz)
        # self.device.set_power(power_dbm)
        print(f" LabWork Frequency {frequency_hz}")
        print(f" LabWork amp {power_dbm}")

        return { "rf_freq": frequency_hz,"rf_amp": power_dbm}



    def transition_to_manual(self, abort=False):
        print(" ===== TRANSITION TO BUFFERED ")
        values = getattr(self, "final_values", None)

        if values is None:
            return True

        frequency_hz = float(values["rf_freq"])
        power_dbm = float(values["rf_amp"])

         # TODO CHECK THESE VALUES 
        # self.device.set_freq(frequency_hz)
        # self.device.set_power(power_dbm)
        print(f" LabWork Frequency {frequency_hz}")
        print(f" LabWork amp {power_dbm}")

        return True

    # ------------------------------------------ Blacs Tabs functions

    def apply_to_device(self, stats : RFGeneratorStats):
        print(stats, type(stats))
        # TODO LATER WHEN READY (PRINT ENOUGH FOR NOW)
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
        # TODO LATER WHEN READY (PRINT ENOUGH FOR NOW)
        # self.device.set_output_rf(state)





