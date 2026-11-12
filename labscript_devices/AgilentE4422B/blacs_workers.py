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

        print("===== PROGRAM MANUAL =====")

        if not front_panel_values:
            return front_panel_values

        print(f"FRONT PANEL VALUES: {front_panel_values}")

        frequency_hz = float(front_panel_values["frequency_hz"])
        power_dbm    = float(front_panel_values["power_dbm"])
        rf_output    = int(front_panel_values["rf_output"])

        # ----------------------------
        # Frequency
        # ----------------------------
        if not np.isnan(frequency_hz):
            print(f"Programming frequency: {frequency_hz} Hz")
            self.device.set_freq(frequency_hz)

        else:
            print("Frequency unchanged")

        # ----------------------------
        # Power
        # ----------------------------
        if not np.isnan(power_dbm):

            print(f"Programming power: {power_dbm} dBm")
            self.device.set_power(power_dbm)

        else:
            print("Power unchanged")

        # ----------------------------
        # RF output state
        # ----------------------------
        if rf_output == -1:
            print("RF output unchanged")

        else:
            rf_enabled = bool(rf_output)
            print(f"Programming RF output: {rf_enabled}")
            self.set_output_rf(rf_enabled)

        return {
            "frequency_hz": frequency_hz,
            "power_dbm": power_dbm,
            "rf_output": rf_output,
        }

        
    def transition_to_buffered(
        self,
        device_name,
        h5_filepath,
        initial_values,
        fresh,
    ):

        print("===== TRANSITION TO BUFFERED =====")

        if getattr(self, "is_remote", False):
            h5_filepath = path_to_local(h5_filepath)

        with h5py.File(h5_filepath, "r") as hdf5_file:

            group = hdf5_file["devices"][device_name]
            shot_values = group["OUTPUT_DATA"][0]

            frequency_hz = float(shot_values["frequency_hz"])
            power_dbm    = float(shot_values["power_dbm"])
            rf_output    = int(shot_values["rf_output"])

        # ----------------------------
        # Frequency
        # ----------------------------
        if not np.isnan(frequency_hz):
            print(f"Programming frequency: {frequency_hz} Hz")
            self.device.set_freq(frequency_hz)

        else:
            print("Frequency unchanged")

        # ----------------------------
        # Power
        # ----------------------------
        if not np.isnan(power_dbm):
            print(f"Programming power: {power_dbm} dBm")
            self.device.set_power(power_dbm)

        else:
            print("Power unchanged")

        # ----------------------------
        # RF output
        # ----------------------------
        if rf_output == -1:

            print("RF output unchanged")

        else:

            rf_enabled = bool(rf_output)

            print(f"Programming RF output: {rf_enabled}")
            self.set_output_rf(rf_enabled)

        return {
            "frequency_hz": frequency_hz,
            "power_dbm": power_dbm,
            "rf_output": rf_output}



    def transition_to_manual(self, abort=False):
        print("===== TRANSITION TO MANUAL =====")
        values = getattr(self, "final_values", None)
        if values is None:
            return True

        frequency_hz = float(values["frequency_hz"])
        power_dbm    = float(values["power_dbm"])
        rf_output    = int(values["rf_output"])

        # ----------------------------
        # Frequency
        # ----------------------------
        if not np.isnan(frequency_hz):
            print(f"Restoring frequency: {frequency_hz} Hz")
            self.device.set_freq(frequency_hz)

        else:
            print("Frequency unchanged")

        # ----------------------------
        # Power
        # ----------------------------
        if not np.isnan(power_dbm):
            print(f"Restoring power: {power_dbm} dBm")
            self.device.set_power(power_dbm)
        else:
            print("Power unchanged")

        # ----------------------------
        # RF output
        # ----------------------------
        if rf_output == -1:
            print("RF output unchanged")
        else:
            rf_enabled = bool(rf_output)
            print(f"Restoring RF output: {rf_enabled}")
            self.set_output_rf(rf_enabled)

        return True
    # ------------------------------------------ Blacs Tabs functions

    def apply_to_device(self, stats : RFGeneratorStats):
        print(stats, type(stats))
        self.device.set_freq(stats.freq_mhz , UnitFreq.MHZ)
        self.device.set_power(stats.power_dbm)
        

    def read_stats_from_device(self) -> RFGeneratorStats:
        freq_read = self.device.get_freq()
        freq_mhz = freq_read / 1e6
        power_dbm = self.device.get_power()
        rf_on = self.device.get_output_rf()
        return RFGeneratorStats(freq_mhz=freq_mhz , power_dbm=power_dbm , rf_on=rf_on)


    def set_output_rf(self, state):
        print(state , type(state))
        self.device.set_output_rf(state)





