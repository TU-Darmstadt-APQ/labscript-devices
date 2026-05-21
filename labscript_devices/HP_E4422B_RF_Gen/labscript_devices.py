from labscript import Device, LabscriptError, set_passed_properties
# --- Intermediate Device Imports 
from labscript_devices import runviewer_parser, BLACS_tab
from labscript_utils.shared_drive import path_to_local
from labscript import config, IntermediateDevice, StaticAnalogQuantity, LabscriptError, set_passed_properties
import numpy as np
import h5py

# --- Blacs Imports 
from blacs.tab_base_classes import define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED
from blacs.device_base_class import DeviceTab
from PyQt5.QtWidgets import QLabel,QWidget,QHBoxLayout
from PyQt5.QtCore import Qt

# --- Worker Imports
from labscript_devices.GPIBDevice import GPIBWorker

# --- Others
# from .logger_config import logger
from agilent_4422B_device import AgilentE4422BDevice, RFGeneratorSpecs, agilent_e4422b_specs



##############################################################################################################
#                                    Intermediate - DEVICE                                                   #
##############################################################################################################
class AgilentE4422B(IntermediateDevice):
    ''' This device controls one RF output (250e3 - 4e9 Hz). Controllable quantities are:
            frequency    # RF carrier frequency in Hz
            power        # RF output power in dBm
            
        Args
            - name (str): Name of the device.
            - GPIB_address (str): GPIB address of the instrument. 
                                - Format: 
                                        1. "GPIB<number>::GPIB_address" 
                                        2. "ADAP::adapter_ip_address::GPIB_address" 
                                - Example:
                                        1. 'GPIB0::5' -> GPIB connection
                                        2. 'ADAP::192.168.123.132::5' -> Adapter connection
            - num_outputs (int): Number of output channels the instrument supports.

        NEW : Depending on the format of the passed GPIB_address : the device will try to connect through:
        -  a socket assuming a KOFOTRONIC adapter (Prologix alike) if GPIB_address = "ADAP::adapter_ip_address::GPIB_address"
        -  pyvisa assuming a normal GPIB connection if fromat GPIB_address =  "GPIB<number>::GPIB_address" 

    '''

    allowed_children = [StaticAnalogQuantity]
    description = 'AgilentE4422B'

    @set_passed_properties()
    def __init__(self, name, GPIB_address , **kwargs):
        IntermediateDevice.__init__(self, name, None, **kwargs)
        self.instructions = {}
        self.BLACS_connection = GPIB_address
        self.specs : RFGeneratorSpecs       = agilent_e4422b_specs 


    # Helper to check RF values
    # TODO Val might be not a signle value
    def _check_output(self, device, output_type):
        val = float(device.static_value) 
        if output_type == "frequency":
            self.specs.validate_frequency(val)
            return True
        elif output_type == "power":
            self.specs.validate_power(val)
            return True
        else:
            raise LabscriptError( f"Output type {output_type!r} is not supported. Use 'frequency'or 'power'.")



    def generate_code(self, hdf5_file):
        IntermediateDevice.generate_code(self, hdf5_file)

        # One static RF output state:
        #   frequency_hz : carrier frequency in Hz
        #   power_dbm    : RF output power in dBm
        dtypes = [ ("frequency_hz", np.float64), ("power_dbm", np.float32)]
        output_table = np.zeros(1, dtype=dtypes)

        expected_connections = {"rf/frequency", "rf/power"}
        for device in self.child_devices:
            
            # Check Device is Static Analog Quantity
            if not isinstance(device, StaticAnalogQuantity):
                raise TypeError(device)
            
            # Check Device connection is in expected_connections
            if device.connection not in expected_connections:
                raise LabscriptError( f"Invalid connection {device.connection}. Use 'rf/frequency' or 'rf/power'.")
            
            # Get the output type
            _, output_type = device.connection.split("/")

            # Check the output value
            self._check_output(device, output_type)


            if output_type == "frequency":
                output_table["frequency_hz"] = float(device.static_value)
            elif output_type == "power":
                output_table["power_dbm"] = float(device.static_value)


        # Create device group in the HDF5 file:
        grp = self.init_device_group(hdf5_file)
        # Save Output to HDF5File:
        grp.create_dataset('OUTPUT_DATA', compression=config.compression, data=output_table)


