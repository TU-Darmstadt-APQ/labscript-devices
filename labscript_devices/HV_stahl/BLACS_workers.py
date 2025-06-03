from blacs.tab_base_classes import Worker
from labscript import LabscriptError
from .logger_config import logger
from datetime import datetime
import h5py
import numpy as np
from zprocess import rich_print
from labscript_utils import properties
from .utils import _ao_to_channel_name, _get_channel_num
import threading
import time

class HV_Worker(Worker):
    def init(self):
        """Initialises communication with the device. When BLACS (re)starts"""
        self.final_values = {}  # [[channel_nums(ints)],[voltages(floats)]]
        self.verbose = True

        self.thread = None
        self._stop_event = threading.Event()
        self._finished_event = threading.Event()

        try:
            # Try to establish a serial connection
            from .high_voltage_source import HighVoltageSource
            self.high_voltage_source = HighVoltageSource(self.port, self.baud_rate)

            # Get device information
            self.device_serial = self.high_voltage_source.device_serial  # For example, 'HV023'
            self.device_voltage_range = self.high_voltage_source.device_voltage_range  # For example, '50'
            self.device_channels = self.high_voltage_source.device_channels  # For example, '10'
            self.device_output_type = self.high_voltage_source.device_output_type  # For example, 'b' (bipolar, unipolar, quadrupole, steerer supply)

            logger.info(
                f"Connected to HV-Series on {self.port} with baud rate {self.baud_rate}\n"
                f"Device Serial: {self.device_serial}, Voltage Range: {self.device_voltage_range}, "
                f"Channels: {self.device_channels}, Output Type: {self.device_output_type}"
            )

        except LabscriptError as e:
            raise RuntimeError(f"HV-series identification failed: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during HV_Worker initialization: {e}")
        
        
    def shutdown(self):
        # Should be done when Blacs is closed
        self.connection.close()
    
    def program_manual(self, front_panel_values): 
        """Allows for user control of the device via the BLACS_tab, 
        setting outputs to the values set in the BLACS_tab widgets. 
        Runs at the end of the shot."""
        rich_print(f"---------- Manual MODE start: ----------", color=BLUE)
        self.front_panel_values = front_panel_values

        if not getattr(self, 'restored_from_final_values', False):
            if self.verbose is True:
                print("Front panel values (before shot):")
                for ch_name, voltage in front_panel_values.items():
                    print(f"  {ch_name}: {voltage:.2f} V")

            # Restore final values from previous shot, if available
            if self.final_values:
                for ch_num, value in self.final_values.items():
                    front_panel_values[f'CH{int(ch_num)}'] = value

            if self.verbose is True:
                print("\nFront panel values (after shot):")
                for ch_num, voltage in self.final_values.items():
                    print(f"  {ch_num}: {voltage:.2f} V")

            self.final_values = {}  # Empty after restoring
            self.restored_from_final_values = True

        return front_panel_values

    def check_remote_values(self): # reads the current settings of the device, updating the BLACS_tab widgets 
        #todo: run at the beginning of program_manual to ensure the programmed values are set correctly.
        # final_values and actual values should be the same.
        return

    def transition_to_buffered(self, device_name, h5_file, initial_values, fresh): 
        """transitions the device to buffered shot mode, 
        reading the shot h5 file and taking the saved instructions from 
        labscript_device.generate_code and sending the appropriate commands 
        to the hardware. 
        Runs at the start of each shot."""
        rich_print(f"---------- Begin transition to Buffered: ----------", color=BLUE)
        self.restored_from_final_values = False  # Drop flag
        self.initial_values = initial_values  # Store the initial values in case we have to abort and restore them
        self.final_values = {}  # Store the final values to update GUI during transition_to_manual
        self.h5file = h5_file  # Store path to h5 to write back from front panel
        self.device_name = device_name

        self._stop_event.clear()
        self._finished_event.clear()

        with h5py.File(h5_file, 'r') as hdf5_file:
            group = hdf5_file['devices'][device_name]
            AO_data = group['AO'][:]

        # prepare events
        events = []
        for row in AO_data:
            t = row['time']
            voltages = {ch: row[ch] for ch in row.dtype.names if ch != 'time'}
            events.append((t, voltages))

        self.thread = threading.Thread(target=self._run_experiment_sequence, args=(events,))
        self.thread.start()

        rich_print(f"---------- End transition to Buffered: ----------", color=BLUE)
        return

    def _run_experiment_sequence(self, events):
        try:
            self._wait_for_trigger()
            if self._stop_event.is_set():
                return

            for t, voltages in events:
                time.sleep(t)
                for conn_name, voltage in voltages.items():
                    channel_num = _get_channel_num(conn_name)  # 'ao1' --> '01'
                    self.high_voltage_source.set_voltage(channel_num, voltage)
                    self.final_values[channel_num] = voltage
                    if self.verbose:
                        print(f"[{t:.3f}s] --> Set {conn_name} (#{channel_num}) = {voltage}")
                    if self._stop_event.is_set():
                        return
        finally:
            self._finished_event.set()
            print(f"[Thread] finished all events !")

    def _wait_for_trigger(self):
        """Wait for external TTL trigger."""
        print("Waiting for trigger...")
        while not self._stop_event.is_set():
            if self._check_trigger():
                print("Trigger received! Start experiment sequence")
                return
            # time.sleep(0.01)
        print("Stopped waiting for trigger.")

    def _check_trigger(self):
        #todo: check if TTL received
        return True

    def abort_transition_to_buffered(self):
        return self.transition_to_manual()

    def transition_to_manual(self): 
        """transitions the device from buffered to manual mode to read/save measurements from hardware
        to the shot h5 file as results. 

        Ensure background thread has finished before exiting the shot."""
        # self._stop_event.set()
        self.thread.join()

        if not self._finished_event.is_set():
            print("Warning: experiment sequence did not finish properly.")
        else:
            print("Experiment sequence completed successfully.")
        return True
    
    def send_to_HV(self, kwargs):
        """Sends manual values from the front panel to the HV device.
            This function is executed in the worker process. It uses the current
            front panel values to reprogram the device in manual mode by clicking the button 'send to device'.
            Args:
                kwargs (dict): Not used currently.
            """
        self._program_manual(self.front_panel_values)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._append_front_panel_values_to_manual(self.front_panel_values, current_time)

    def _program_manual(self, front_panel_values):
        """Sends voltage values to the device for all channels using HighVoltageSource.

           Parameters:
           - front_panel_values (dict): Dictionary of voltages keyed by channel name (e.g., 'CH1', 'CH2', ...).
           """
        if self.verbose:
            print("\nProgramming the device with the following values:")
            logger.info("Programming the device from manual with the following values:")

        for channel_num in range(1, int(self.num_AO) + 1):
            channel_name = f'CH{channel_num}' # 'CH1'
            try:
                voltage = front_panel_values[channel_name]
            except Exception as e:
                raise ValueError(f"Error accessing front panel values for channel '{channel_name}': {e}")

            if self.verbose:
                print(f"→ {channel_name}: {voltage:.2f} V")
            logger.info(f"Setting {channel_name} to {voltage:.2f} V (manual mode)")

            self.high_voltage_source.set_voltage(channel_num, voltage)

    def _append_front_panel_values_to_manual(self, front_panel_values, current_time):
        """
            Append front-panel voltage values to the 'AO_manual' dataset in the HDF5 file.

            This method records the current manual voltage settings (from the front panel)
            along with a timestamp into the 'AO_manual' table inside the device's HDF5 group.
            It assumes that `self.h5file` and `self.device_name` have been set
            (in `transition_to_buffered`). If not, a RuntimeError is raised.

            Parameters
            ----------
            front_panel_values : dict
                Dictionary mapping channel names (e.g., 'channel 0') to voltage values (float).
            current_time : str
                The timestamp (formatted as a string) when the values were recorded

            Raises
            ------
            RuntimeError
                If `self.h5file` is not set (i.e., manual values are being saved before
                the system is in buffered mode).
            """
        # Check if h5file is set (transition_to_buffered must be called first)
        if not hasattr(self, 'h5file') or self.h5file is None:
            raise RuntimeError(
                "Cannot save manual front-panel values: "
                "`self.h5file` is not set. Make sure `transition_to_buffered()` has been called before sending to the device."
            )

        with h5py.File(self.h5file, 'r+') as hdf5_file:
            group = hdf5_file['devices'][self.device_name]

            dset = group['AO_manual']
            old_shape = dset.shape[0]
            dtype = dset.dtype
            connections = [name for name in dset.dtype.names if name != 'time'] #'ao1'

            # Create new data row
            new_row = np.zeros((1,), dtype=dtype)
            new_row['time'] = current_time
            for conn in connections: # 'ao0'
                channel_name = _ao_to_channel_name(conn) # 'ao0' --> 'CH1'
                new_row[conn] = front_panel_values.get(channel_name, 0.0)

            # Add new row to table
            dset.resize(old_shape + 1, axis=0)
            dset[old_shape] = new_row[0]



# --------------------contants
PINK = 'ff52fa'
BLUE = '#66D9EF'