from labscript_devices import register_classes
from labscript import (
    IntermediateDevice,
    AnalogOut,
    DigitalOut,
    StaticAnalogOut,
    StaticDigitalOut,
    AnalogIn,
    bitfield,
    config,
    compiler,
    LabscriptError,
    set_passed_properties,
)
from .utils import split_conn_AO
import numpy as np
import h5py
from .logger_config import logger

class HV_(IntermediateDevice):
    description = 'HV_Series' # take from models cap
    
    #@set_passed_properties({"connection_table_properties": ["port", "baud_rate", "num_AO"]})
    @set_passed_properties(
        property_names={
            "connection_table_properties": [
                "port",
                "baud_rate",
                "AO_range",
                "num_AO",
                "static_AO",
            ],
            "device_properties": ["num_AO"],
        }
    )
    def __init__(self, name, port='', baud_rate=9600, parent_device=None, num_AO=0, AO_range=0, static_AO=None, **kwargs):
        """Generic class for NI_DAQmx devices.

        Generally over-ridden by device-specific subclasses that contain
        the introspected default values.

        Args:
            name (str): name to assign to the created labscript device
            parent_device (clockline): Parent clockline device that will
                clock the outputs of this device
            AO_range (iterable, optional): A `[Vmin, Vmax]` pair that sets the analog
                output voltage range for all analog outputs.
            static_AO (int, optional): Number of static analog output channels.
            num_AO (int, optional): Number of analog output channels.
        """
        self.num_AO = num_AO
        self.AO_range = AO_range
        self.static_AO = static_AO
        IntermediateDevice.__init__(self, name, parent_device, **kwargs)
        self.BLACS_connection = '%s,%s' % (port, str(baud_rate))
    
    def add_device(self, device):
        IntermediateDevice.add_device(self, device)

    def generate_code(self, hdf5_file):
        """Convert the list of commands into numpy arrays and save them to the shot file."""
        logger.info("generate_code for HV-Series is called")
        IntermediateDevice.generate_code(self, hdf5_file)

        clockline = self.parent_device
        pseudoclock = clockline.parent_device
        times = pseudoclock.times[clockline]

        # create dataset
        analogs = {}
        for child_device in self.child_devices:
            if isinstance(child_device, AnalogOut):
                analogs[child_device.connection] = child_device

        AO_table = self._make_analog_out_table(analogs, times)
        AO_manual_table = self._make_analog_out_table_from_manual(analogs)
        logger.info(f"Times in generate_code AO table: {times}")
        logger.info(f"AO table for HV-Series is: {AO_table}")

        group = self.init_device_group(hdf5_file)
        group.create_dataset("AO", data=AO_table, compression=config.compression)
        group.create_dataset("AO_manual", shape=AO_manual_table.shape, maxshape=(None,), dtype=AO_manual_table.dtype,
                             compression=config.compression, chunks=True)

    def _make_analog_out_table(self, analogs, times):
        """Collect analog output data and create the output table"""
        if not analogs:
            return None
        n_timepoints = 1 if (self.static_AO is not None) else len(times)
        connections = sorted(analogs, key=split_conn_AO)
        dtypes = [('time', np.float64)] + [(c, np.float32) for c in connections] # first column is time ('t' from seq. logic)
        analog_out_table = np.empty(n_timepoints, dtype=dtypes)
        analog_out_table['time'] = times
        for connection, output in analogs.items():
            analog_out_table[connection] = output.raw_output
        return analog_out_table

    def _make_analog_out_table_from_manual(self, analogs):
        """Create a structured empty numpy array with first column as 'time', followed by analog channel data.
        Args:
            times (array-like): Array of timestamps.
            ...
        Returns:
            np.ndarray: Structured empty array with time and analog outputs."""

        str_dtype = h5py.string_dtype(encoding='utf-8', length=19)

        connections = sorted(analogs, key=split_conn_AO)  # sorted channel names
        dtypes = [('time', str_dtype)] + [(c, np.float32) for c in connections]

        analog_out_table = np.empty(0, dtype=dtypes)
        return analog_out_table