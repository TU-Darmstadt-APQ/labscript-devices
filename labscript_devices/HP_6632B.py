from labscript_devices import runviewer_parser, BLACS_tab
from labscript_utils.shared_drive import path_to_local
from labscript import config, Device, IntermediateDevice, StaticAnalogQuantity, LabscriptError, set_passed_properties
import numpy as np
import labscript_utils.h5_lock
import labscript_utils.properties
import h5py

# Specifications for HP6632B:
max_no_of_outputs = 1
Watt_ratings = [100]
voltage_decimals = 3
current_decimals = 4

# DC Output Range Specifications
LOW_RANGE = False
MIN_VOLTAGE = 0
MAX_VOLTAGE = 20
MIN_CURRENT = 0
MAX_CURRENT = 5

# Descriptions:

# This Devices allows 4 "StaticAnalogQuantity"s to be set: Voltage and Current for each of the two outputs.
# Each Quantity is named by its connections similar to the NI_DAQmx principle using: outX/voltage or outX/current
# for voltage and current values of the outputs.


# TODOs
# Check LOW_RANGE and HIGH_RANGE for each connection! At the moment general limit defined by MIN_VOLTAGE, etc.
# if isinstance(self.LOW_RANGE, bool):
#           self.LOW_RANGE = LOW_RANGE
#       else:
#           raise TypeError(self.LOW_RANGE)


class HP_6632B(IntermediateDevice):
    allowed_children = [StaticAnalogQuantity]

    description = 'HP 6632B DC Power Supply'

    @set_passed_properties(property_names={"connection_table_properties": ["num_outputs"]})
    def __init__(self, name, GPIB_address, num_outputs=None, **kwargs):
        # Following Phil's thesis, IntermediateDevice should be subclassed here:
        IntermediateDevice.__init__(self, name, None, **kwargs)

        self.instructions = {}

        self.BLACS_connection = GPIB_address
        if isinstance(num_outputs, int):
            if num_outputs <= max_no_of_outputs:
                self.num_outputs = num_outputs
            else:
                raise Exception("A maximum of {:f} outputs is allowed.".format(max_no_of_outputs))
        elif isinstance(num_outputs, None):
            raise Exception('Please specify the number of used outputs in the connection table')
        else:
            raise TypeError()

    # def add_device(self, output):
    #     # This device has 2 valid connection-ports: "freqeuncy" and "range"
    #     # range may be disabled
    #     for dev in self.child_devices:
    #         if output.connection == dev.connection:
    #             # e.g.: only one frequency AnalogQuantitiy is allowed
    #             raise LabscriptError("Cannot add more than 1 output to {:s} on the same connection {:s}".format(self.name, output.connection))
    #     if not output.connection in ("frequency", "range"):
    #         raise LabscriptError("Cannot attach '{:s}' to '{:s}'. The connection must be a valid value. ({:s})".format(output.name, self.name, "frequency, range"))
    #     if not ENABLE_RANGE and output.connection == "range":
    #         raise LabscriptError("Cannot attach '{:s}' to '{:s}' on port 'range'. Range is disabled.".format(output.name, self.name))
    #     IntermediateDevice.add_device(self, output)

    def generate_code(self, hdf5_file):
        IntermediateDevice.generate_code(self, hdf5_file)

        # do checks
        # Initialise output table as structered numpy array with 8 fields named 'vi' and 'ci' where i=1...4 is the channel number.
        # output_voltage = {}
        # output_current = {}

        dtypes = [('v%d' % (i + 1), np.float32) for i in range(4)] + \
                 [('c%d' % (i + 1), np.float32) for i in range(4)]
        # print('dtypes',dtypes)
        output_table = np.zeros(1, dtype=dtypes)
        # print('output_table',output_table)

        # iterate through the connected child devices
        # Child devices are the StaticAnalogQuantities connected
        # TODO: Improve Range check! for each Output and each possible range!

        for device in self.child_devices:
            try:
                channel_no, output_type = device.connection.replace('out', '').split('/')
                output_table['%s%d' % (output_type[0], int(channel_no))] = device.static_value
            except (ValueError, IndexError):
                msg = """Connection string %s does not match format 'out<N>/voltage' or 'out<N>/current' for integer N"""
                raise ValueError(msg % str(device.connection))

        # Check for device-specific limits
        for i in range(2):
            i += 1
            if output_table['v%d' % i] < MIN_VOLTAGE or output_table['v%d' % i] > MAX_VOLTAGE:
                raise LabscriptError("The voltage specified for {:s} is not within the power supply's voltage range".format(device.connection))
        for i in range(2):
            i += 1
            if output_table['c%d' % i] < MIN_CURRENT or output_table['c%d' % i] > MAX_CURRENT:
                raise LabscriptError("The voltage specified for {:s} is not within the power supply's voltage range".format(device.connection))

        # print('output_table',output_table)

        # Create device group in the HDF5 file:
        grp = self.init_device_group(hdf5_file)

        # Save Output to HDF5File:
        grp.create_dataset('OUTPUT_DATA', compression=config.compression, data=output_table)


import os
from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED

from blacs.device_base_class import DeviceTab

from qtutils import UiLoader
from qtutils.qt.QtCore import *
from qtutils.qt.QtGui import QDoubleValidator


@BLACS_tab
class HP_6632BTab(DeviceTab):

    def initialise_GUI(self):
        # Pull the following information out of the connection table:
        connection_table = self.settings['connection_table']
        connection_table_properties = connection_table.find_by_name(self.device_name).properties
        self.num_outputs = 1  # connection_table_properties['num_outputs']

        layout = self.get_tab_layout()

        # Capabilities:

        self.base_units = {'v': 'V', 'c': 'A'}
        self.base_step = {'v': 0.1, 'c': 0.01}  # step size for +/- buttons
        self.base_decimals = {'v': voltage_decimals, 'c': current_decimals}  # display 2 decimals accuracy

        analog_properties = {}
        for i in range(2):
            analog_properties['out%d/voltage' % (i + 1)] = {'base_unit': self.base_units['v'],
                                                            'min': MIN_VOLTAGE,
                                                            'max': MAX_VOLTAGE,
                                                            'step': self.base_step['v'],
                                                            'decimals': self.base_decimals['v']
                                                            }
            analog_properties['out%d/current' % (i + 1)] = {'base_unit': self.base_units['c'],
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
        self.create_worker("main_worker", HP_6632BWorker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"


from labscript_devices.GPIBDevice import GPIBWorker


class HP_6632BWorker(GPIBWorker):

    # TODO: Check that if no voltage is specified, nothing is sent.
    # TODO: Include programming accuracies
    def send_GPIB_voltage(self, voltage=None, output=None):
        # Update the power supply outputs with the specified voltages.
        # If an argument is None, the corresponding value will not be changed
        if voltage is not None:
            voltage = np.round(voltage, voltage_decimals)  # round voltage to four decimal places!

            if voltage < MIN_VOLTAGE or voltage > MAX_VOLTAGE:
                raise Exception("Voltage {:f} is out of range {:f} to {:f}. Is the voltage in V?".format(voltage, MIN_VOLTAGE, MAX_VOLTAGE))

        if voltage is not None and output is not None:
            sendStr = "VOLT "
            # sendStr += str(output)
            # sendStr += ","
            sendStr += str(voltage)
            self.GPIB_connection.write(sendStr)
            # print('send', sendStr)

    def send_GPIB_current(self, current=None, output=None):
        # Update the power supply  current outputs with the specified currents.
        # If an argument is None, the corresponding value will not be changed
        if current is not None:
            current = np.round(current, current_decimals)  # round current to three decimal places!

            if current < MIN_CURRENT or current > MAX_CURRENT:
                raise Exception("Current {:f} is out of range {:f} to {:f}. Is the Current in A?".format(current, MIN_CURRENT, MAX_CURRENT))

        if current is not None and output is not None:
            sendStr = "CURR "
            # sendStr += str(output)
            # sendStr += ","
            sendStr += str(current)
            self.GPIB_connection.write(sendStr)
            # print('send', sendStr)

    # TODO: check for remote values and warn if control_mode is changing
    def check_channel_control(self, channel):
        return
        # out_value_voltage = np.round(np.float(self.GPIB_connection.query('VOUT?')), voltage_decimals)
        # out_value_current = np.round(np.float(self.GPIB_connection.query('IOUT?')), current_decimals)

        # print('Output voltage is {}; output current is {}'.format(out_value_voltage, out_value_current))

    def check_remote_values(self):
        for i in range(self.num_outputs):
            self.check_channel_control(i + 1)

    def program_manual(self, front_panel_values):
        # Get values from the front_panel_settings
        for i in range(self.num_outputs):
            voltage = front_panel_values['out' + str(i + 1) + '/voltage']
            self.send_GPIB_voltage(voltage=voltage, output=i + 1)

        for i in range(self.num_outputs):
            current = front_panel_values['out' + str(i + 1) + '/current']
            self.send_GPIB_current(current=current, output=i + 1)
        self.check_remote_values()
        return {}  # no need to adjust the values. Can add a check_remote_values() here to read current values from power supply

    def transition_to_buffered(self, device_name, h5_filepath, initial_values, fresh):
        # for remote worker to find correct find path:
        if getattr(self, 'is_remote', False):
            h5_filepath = path_to_local(h5_filepath)

        # Get values at first from 'initial_values' and overwrite them afterwards with values given in the experiment script
        dtypes = [('v%d' % (i + 1), np.float32) for i in range(4)] + \
                 [('c%d' % (i + 1), np.float32) for i in range(4)]
        # print('dtypes',dtypes)
        output_table = np.zeros(1, dtype=dtypes)
        for i in range(self.num_outputs):
            output_table['v%d' % (i + 1)] = initial_values['out' + str(i + 1) + '/voltage']
            output_table['c%d' % (i + 1)] = initial_values['out' + str(i + 1) + '/current']

        # Get values from experiment script
        with h5py.File(h5_filepath, 'r') as hdf5_file:
            group = hdf5_file['devices'][device_name]
            output_table = group['OUTPUT_DATA'][0]

        # Send Values via GPIB:
        final_values = {}
        for i in range(self.num_outputs):
            self.send_GPIB_voltage(voltage=output_table['v%d' % (i + 1)], output=i + 1)
            final_values['out' + str(i + 1) + '/voltage'] = output_table['v%d' % (i + 1)]
            self.send_GPIB_current(current=output_table['c%d' % (i + 1)], output=i + 1)
            final_values['out' + str(i + 1) + '/current'] = output_table['c%d' % (i + 1)]
        # Return final values to use them when transitioning to manual:
        self.final_values = final_values
        return self.final_values

    def transition_to_manual(self, abort=False):
        # Set all channels to their final values:
        values = self.final_values

        voltage_table = np.empty(self.num_outputs)
        current_table = np.empty(self.num_outputs)
        for i in range(self.num_outputs):
            voltage_table[i] = values['out' + str(i + 1) + '/voltage']
        for i in range(self.num_outputs):
            current_table[i] = values['out' + str(i + 1) + '/current']

        for i in range(len(voltage_table)):
            self.send_GPIB_voltage(voltage=voltage_table[i], output=i + 1)
        for i in range(len(current_table)):
            self.send_GPIB_current(current=current_table[i], output=i + 1)

        # return True to indicate we successfully transitioned back to manual mode
        return True


@runviewer_parser
class HP_6632BParser(object):
    pass
# class RunviewerClass(object):

    # def __init__(self, path, device):
    #     self.path = path
    #     self.name = device.name
    #     self.device = device

    # def get_traces(self, add_trace, clock=None):
    #     # the clock argument is used as stop time, because this device is not connected to any Masterclock
    #     if not (isinstance(clock, float) or isinstance(clock, int)):  # shoud never happen
    #         raise Exception("No stop time is passed to RS SignalGenerator")
    #     else:
    #         stop_time = clock

    #     # get the shot data
    #     with h5py.File(self.path, 'r') as f:
    #         if 'FREQUENCY_OUTPUT' in f['devices/%s' % self.name]:
    #             data = f['devices/%s/FREQUENCY_OUTPUT' % self.name][:]
    #             frequency = data[0][0]
    #         else:
    #             frequency = None

    #     traces = {}
    #     traces['frequency'] = ((0, stop_time), (frequency, frequency))  # add start- and end point to the trace

    #     triggers = {}
    #     for channel_name, channel in self.device.child_list.items():
    #         if channel.parent_port in traces:
    #             if channel.device_class == 'Trigger':
    #                 triggers[channel_name] = traces[channel.parent_port]
    #             add_trace(channel_name, traces[channel.parent_port], self.name, channel.parent_port)

    #     return triggers
