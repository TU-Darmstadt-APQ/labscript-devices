# --- Intermediate Device Imports 
from labscript_devices import runviewer_parser, BLACS_tab
from labscript_utils.shared_drive import path_to_local
from labscript import config, Device, IntermediateDevice, StaticAnalogQuantity, LabscriptError, set_passed_properties
import numpy as np
import labscript_utils.h5_lock
import labscript_utils.properties
import h5py

# --- Blacs Imports 
import os
from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED
from blacs.device_base_class import DeviceTab

from qtutils import UiLoader
from qtutils.qt.QtCore import *
from qtutils.qt.QtGui import QDoubleValidator

# --- Worker Imports
from labscript_devices.GPIBDevice import GPIBWorker


# --- Specifications for HP6632B:
max_no_of_outputs = 1
Watt_ratings = [100]
voltage_decimals = 3
current_decimals = 4

# --- DC Output Range Specifications
LOW_RANGE = False
MIN_VOLTAGE = 0
MAX_VOLTAGE = 20
MIN_CURRENT = 0
MAX_CURRENT = 5


class HP_6632A(IntermediateDevice):

    '''
    NEW : By passsing a value for Ip_adapter, the device will try to connect through a KOFOTRONIC adapter (Prologix alike)
    '''
    allowed_children = [StaticAnalogQuantity]
    description = 'HP 6632A DC Power Supply'

    @set_passed_properties(property_names={"connection_table_properties": ["ip_adapter","num_outputs"]})
    def __init__(self, 
                 name, 
                 GPIB_address, 
                 ip_adapter= None,
                 num_outputs=None,
                   **kwargs):
        IntermediateDevice.__init__(self, name, None, **kwargs)

        self.instructions = {}
        self.BLACS_connection = GPIB_address
        self.ip_adapter = ip_adapter

        if isinstance(num_outputs, int):
            if num_outputs <= max_no_of_outputs:
                self.num_outputs = num_outputs
            else:
                raise Exception("A maximum of {:f} outputs is allowed.".format(max_no_of_outputs))
        elif isinstance(num_outputs, None):
            raise Exception('Please specify the number of used outputs in the connection table')
        else:
            raise TypeError()


    def generate_code(self, hdf5_file):
        IntermediateDevice.generate_code(self, hdf5_file)


        dtypes = [('v', np.float32)] + \
                 [('c', np.float32)]

        output_table = np.zeros(1, dtype=dtypes)

        for device in self.child_devices:
            try:
                output_type = device.connection.replace('out/', '')
                output_table['%s' % (output_type[0])] = device.static_value
            except (ValueError, IndexError):
                msg = """Connection string %s does not match format 'out<N>/voltage' or 'out<N>/current' for integer N"""
                raise ValueError(msg % str(device.connection))


        for i in range(2):
            i += 1
            if output_table['v'] < MIN_VOLTAGE or output_table['v'] > MAX_VOLTAGE:
                raise LabscriptError("The voltage specified for {:s} is not within the power supply's voltage range".format(device.connection))
        for i in range(2):
            i += 1
            if output_table['c'] < MIN_CURRENT or output_table['c'] > MAX_CURRENT:
                raise LabscriptError("The voltage specified for {:s} is not within the power supply's voltage range".format(device.connection))

        grp = self.init_device_group(hdf5_file)

        grp.create_dataset('OUTPUT_DATA', compression=config.compression, data=output_table)



@BLACS_tab
class HP_6632ATab(DeviceTab):

    def initialise_GUI(self):

        connection_table = self.settings['connection_table']
        connection_table_properties = connection_table.find_by_name(self.device_name).properties
        self.num_outputs = connection_table_properties["num_outputs"]

        # layout = self.get_tab_layout()

        # Capabilities:

        self.base_units = {'v': 'V', 'c': 'A'}
        self.base_step = {'v': 0.1, 'c': 0.01}  # step size for +/- buttons
        self.base_decimals = {'v': voltage_decimals, 'c': current_decimals}  # display 2 decimals accuracy

        analog_properties = {}
        analog_properties['out/voltage'] = {'base_unit': self.base_units['v'],
                                            'min': MIN_VOLTAGE,
                                            'max': MAX_VOLTAGE,
                                            'step': self.base_step['v'],
                                            'decimals': self.base_decimals['v']
                                            }
        analog_properties['out/current'] = {'base_unit': self.base_units['c'],
                                            'min': MIN_CURRENT,
                                            'max': MAX_CURRENT,
                                            'step': self.base_step['c'],
                                            'decimals': self.base_decimals['c']
                                            }

        self.create_analog_outputs(analog_properties)

        _, ao_widgets, _ = self.auto_create_widgets()

        self.auto_place_widgets(ao_widgets)

        connection_table_entry = self.settings['connection_table'].find_by_name(self.settings['device_name'])
        self.GPIB_address = connection_table_entry.BLACS_connection

    @define_state(MODE_MANUAL, True)
    def transition_to_buffered(self, h5_filepath, notify_queue):
        # for remote worker to find correct find path:
        if getattr(self, 'is_remote', False):
            h5_filepath = path_to_local(h5_filepath)
        DeviceTab.transition_to_buffered(self, h5_filepath, notify_queue)

    @define_state(MODE_BUFFERED, False)
    def transition_to_manual(self, notify_queue, program=False):
        DeviceTab.transition_to_manual(self, notify_queue, program)

    def initialise_workers(self):
        worker_initialisation_kwargs = {'GPIB_address': self.GPIB_address, 'num_outputs': self.num_outputs}
        self.create_worker("main_worker", HP_6632AWorker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"



class HP_6632AWorker(GPIBWorker):

    def init(self):


        # ----------------------------------------- Initialize osci
        global HP_6632A_Device
        self.hp = HP_6632A_Device(
            GPIB_address = self.address,
            )

    def send_GPIB_voltage(self, voltage=None):
        if voltage is not None:
            voltage = np.round(voltage, voltage_decimals) 

            if voltage < MIN_VOLTAGE or voltage > MAX_VOLTAGE:
                raise Exception("Voltage {:f} is out of range {:f} to {:f}. Is the voltage in V?".format(voltage, MIN_VOLTAGE, MAX_VOLTAGE))

        if voltage is not None:
            sendStr = "VOLT "
            sendStr += str(voltage)
            self.GPIB_connection.write(sendStr)

    def send_GPIB_current(self, current=None):
        if current is not None:
            current = np.round(current, current_decimals)  

            if current < MIN_CURRENT or current > MAX_CURRENT:
                raise Exception("Current {:f} is out of range {:f} to {:f}. Is the Current in A?".format(current, MIN_CURRENT, MAX_CURRENT))

        if current is not None:
            sendStr = "CURR "
            sendStr += str(current)
            self.GPIB_connection.write(sendStr)


    def check_channel_control(self, channel):
        return

    def check_remote_values(self):
        pass
        # for i in range(self.num_outputs):
        #     self.check_channel_control(i + 1)

    def program_manual(self, front_panel_values):
        # for i in range(self.num_outputs):
        #     voltage = front_panel_values['out/voltage']
        #     self.send_GPIB_voltage(voltage=voltage)

        # for i in range(self.num_outputs):
        #     current = front_panel_values['out/current']
        #     self.send_GPIB_current(current=current)
        # self.check_remote_values()
        return {}
    
    def transition_to_buffered(self, device_name, h5_filepath, initial_values, fresh):
        # if getattr(self, 'is_remote', False):
        #     h5_filepath = path_to_local(h5_filepath)

        # dtypes = [('v', np.float32)] + \
        #          [('c', np.float32)]
        
        # output_table = np.zeros(1, dtype=dtypes)
        # for i in range(self.num_outputs):
        #     output_table['v'] = initial_values['out/voltage']
        #     output_table['c'] = initial_values['out/current']

        # with h5py.File(h5_filepath, 'r') as hdf5_file:
        #     group = hdf5_file['devices'][device_name]
        #     output_table = group['OUTPUT_DATA'][0]

        # final_values = {}
        # for i in range(self.num_outputs):
        #     self.send_GPIB_voltage(voltage=output_table['v'])
        #     final_values['out/voltage'] = output_table['v']
        #     self.send_GPIB_current(current=output_table['c'])
        #     final_values['out/current'] = output_table['c']
        # self.final_values = final_values
        # return self.final_values
        return {}

    def transition_to_manual(self, abort=False):
        # values = self.final_values
        # voltage_table = np.empty(self.num_outputs)
        # current_table = np.empty(self.num_outputs)
        # voltage_table[0] = values['out/voltage']
        # current_table[0] = values['out/current']

        # self.send_GPIB_voltage(voltage=voltage_table[0])
        # self.send_GPIB_current(current=current_table[0])
        return True


class HP_6632A_Device:
    def __init__(self,GPIB_address):
        self.GPIB_address = GPIB_address
        print(self.GPIB_address)
