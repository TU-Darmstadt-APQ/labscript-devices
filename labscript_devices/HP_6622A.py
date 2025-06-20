# -------------------------------- Imports
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

# --- Worker Imports
from labscript_devices.GPIBDevice import GPIBWorker

# --- Others
from .logger_config import logger

##############################################################################################################
#                                               Device Limits                                                #
##############################################################################################################

# --- Specifications for HP6622A:
max_no_of_outputs : int  = 2
Watt_ratings : list[int] = [80, 80]
voltage_decimals : int   = 2
current_decimals : int   = 3

# --- DC Output Range Specifications
LOW_RANGE : bool = False
MIN_VOLTAGE = 0
MAX_VOLTAGE = 20
MIN_CURRENT = 0
MAX_CURRENT = 4

MIN_VOLTAGE_80W_LOW_RANGE = 0  # in V; Outputs 1 and 2
MAX_VOLTAGE_80W_LOW_RANGE = 20  # in V; Outputs 1 and 2

MIN_CURRENT_80W_LOW_RANGE = 0  # in A; Outputs 1 and 2
MAX_CURRENT_80W_LOW_RANGE = 4  # in A; Outputs 1 and 2


MIN_VOLTAGE_80W_HIGH_RANGE = 0  # in V; Outputs 1 and 2
MAX_VOLTAGE_80W_HIGH_RANGE = 50  # in V; Outputs 1 and 2

MIN_CURRENT_80W_HIGH_RANGE = 0  # in A; Outputs 1 and 2
MAX_CURRENT_80W_HIGH_RANGE = 2  # in A; Outputs 1 and 2


##############################################################################################################
#                                               DEVICE                                                       #
##############################################################################################################

class HP_6622A(IntermediateDevice):
    '''
        This Devices allows 4 "StaticAnalogQuantity"s to be set: Voltage and Current for each of the two outputs.
        Each Quantity is named by its connections similar to the NI_DAQmx principle using: outX/voltage or outX/current
        for voltage and current values of the outputs.

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
    description = 'HP 6622A DC Power Supply'

    @set_passed_properties(property_names={"connection_table_properties": ["num_outputs"]})
    def __init__(self, name, GPIB_address, num_outputs=None, **kwargs):
        IntermediateDevice.__init__(self, name, None, **kwargs)

        self.instructions = {}
        self.BLACS_connection = GPIB_address

        # Check the number of outputs
        if isinstance(num_outputs, int):
            if num_outputs <= max_no_of_outputs:
                self.num_outputs = num_outputs
            else:
                raise Exception("A maximum of {:f} outputs is allowed.".format(max_no_of_outputs))
        elif isinstance(num_outputs, None):
            raise Exception('Please specify the number of used outputs in the connection table')
        else:
            raise TypeError()
        
    def _check_output(self, device, output_type):
        val = float(device.static_value)
        if output_type == "voltage":
            if val < MIN_VOLTAGE_80W_HIGH_RANGE or val > MAX_VOLTAGE_80W_HIGH_RANGE:
                return False
            return True
        elif output_type == "current":
            if val < MIN_CURRENT_80W_HIGH_RANGE or val > MAX_CURRENT_80W_HIGH_RANGE:
                return False
            return True
        else:
            raise LabscriptError(f" output type {output_type} is not supported. Only 'voltage' and 'current.")

    def generate_code(self, hdf5_file):
        IntermediateDevice.generate_code(self, hdf5_file)

        # Initialise output table as structered numpy array with fields named 'vi' and 'ci', where 1 ≤ i ≤ num_outputs.
        # output_voltage, output_current = {}, {}
        dtypes = [('v%d' % (i + 1), np.float32) for i in range(self.num_outputs)] + \
                 [('c%d' % (i + 1), np.float32) for i in range(self.num_outputs)]

        # Check connected child devices and create the output table and the analogs dictionary
        output_table = np.zeros(1, dtype=dtypes)
        analogs = {}
        for device in self.child_devices:
            try:
                if isinstance(device, StaticAnalogQuantity):
                    analogs[device.connection] = device
                else: raise TypeError(device)

                channel_no, output_type = device.connection.replace('out', '').split('/')
                logger.info(f"output_type {output_type}")
                logger.info(f"device {type(device)}")
                if not self._check_output( device , output_type):
                    raise LabscriptError(f"The {output_type} specified for {device.connection} is not within the power supply's {output_type} range")

                output_table['%s%d' % (output_type[0], int(channel_no))] = device.static_value

            except (ValueError, IndexError):
                raise ValueError(f"Connection string {device.connection} does not match format 'out<N>/voltage' or 'out<N>/current' for integer N. The Error is {e}")

        # Create device group in the HDF5 file:
        grp = self.init_device_group(hdf5_file)

        # Save Output to HDF5File:
        grp.create_dataset('OUTPUT_DATA', compression=config.compression, data=output_table)



##############################################################################################################
#                                                TAB                                                         #
##############################################################################################################

@BLACS_tab
class HP_6622ATab(DeviceTab):

    def initialise_GUI(self):
        # Pull the following information out of the connection table:
        connection_table = self.settings['connection_table']
        connection_table_properties = connection_table.find_by_name(self.device_name).properties
        self.num_outputs = int(connection_table_properties['num_outputs'])

        # Capabilities:
        self.base_units = {'v': 'V', 'c': 'A'}
        self.base_step = {'v': 0.1, 'c': 0.01}  # step size for +/- buttons
        self.base_decimals = {'v': voltage_decimals, 'c': current_decimals}  # display accuracy

        analog_properties = {}
        for i in range(self.num_outputs):
            analog_properties['out%d/voltage' % (i + 1)] = {'base_unit': self.base_units['v'],
                                                            'min': MIN_VOLTAGE_80W_HIGH_RANGE,
                                                            'max': MAX_VOLTAGE_80W_HIGH_RANGE,
                                                            'step': self.base_step['v'],
                                                            'decimals': self.base_decimals['v']
                                                            }
            analog_properties['out%d/current' % (i + 1)] = {'base_unit': self.base_units['c'],
                                                            'min': MIN_CURRENT_80W_HIGH_RANGE,
                                                            'max': MAX_CURRENT_80W_HIGH_RANGE,
                                                            'step': self.base_step['c'],
                                                            'decimals': self.base_decimals['c']
                                                            }

        self.create_analog_outputs(analog_properties)

        _, ao_widgets, _ = self.auto_create_widgets()
        # logger.info(f"{ao_widgets}")

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
        self.create_worker("main_worker", HP_6622AWorker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"



##############################################################################################################
#                                                WORKER                                                      #
##############################################################################################################

class HP_6622AWorker(GPIBWorker):

    # -------------------------- Instrument specific methodes
    def set_v(self,chan,voltage):
        try:
            sendStr = f"VSET {chan},{voltage}"
            self.GPIB_connection.write(sendStr)
        except Exception as e:
            raise LabscriptError("Failed setting voltage : ",e)

    def get_v(self,chan):
        try:
            sendStr = f"VOUT? {chan}" 
            return self.GPIB_connection.query(sendStr)
        except Exception as e:
            raise LabscriptError("Failed getting voltage : ",e)
        
    def set_i(self, chan, current):
        try:
            sendStr = f"ISET {chan},{current}"
            self.GPIB_connection.write(sendStr)
        except Exception as e:
            raise LabscriptError(f"Failed setting current: {e}")

    def get_i(self, chan):
        try:
            sendStr = f"IOUT? {chan}"
            return self.GPIB_connection.query(sendStr)
        except Exception as e:
            raise LabscriptError(f"Failed getting current: {e}")

    # -------------------------- Worker methodes
    # TODO: Check that if no voltage is specified, nothing is sent.
    # TODO: Include programming accuracies
    def send_GPIB_voltage(self, voltage=None, output=None):
        # Update the power supply outputs with the specified voltages.
        # If an argument is None, the corresponding value will not be changed
        if voltage is not None:
            voltage = np.round(voltage, voltage_decimals) 

            if voltage < MIN_VOLTAGE or voltage > MAX_VOLTAGE:
                raise Exception("Voltage {:f} is out of range {:f} to {:f}. Is the voltage in V?".format(voltage, MIN_VOLTAGE, MAX_VOLTAGE))

        if voltage is not None and output is not None:
            self.set_v(output,voltage)

    def send_GPIB_current(self, current=None, output=None):
        # Update the power supply current outputs with the specified currents.
        # If an argument is None, the corresponding value will not be changed
        if current is not None:
            current = np.round(current, current_decimals)  # round current to three decimal places!

            if current < MIN_CURRENT or current > MAX_CURRENT:
                raise Exception("Current {:f} is out of range {:f} to {:f}. Is the Current in A?".format(current, MIN_CURRENT, MAX_CURRENT))

        if current is not None and output is not None:
            self.set_i(output,current)


    # TODO: check for remote values and warn if control_mode is changing
    def check_channel_control(self, channel):
        set_value_voltage = np.round(float(self.GPIB_connection.query('VSET?' + str(channel))), voltage_decimals)
        set_value_current = np.round(float(self.GPIB_connection.query('ISET?' + str(channel))), current_decimals)
        out_value_voltage = np.round(float(self.GPIB_connection.query('VOUT?' + str(channel))), voltage_decimals)
        out_value_current = np.round(float(self.GPIB_connection.query('IOUT?' + str(channel))), current_decimals)

        if out_value_current >= set_value_current:
            print('Channel %d is in Current Control mode, Out: %.4f, Set: %.4f' % (channel, out_value_current, set_value_current))
        elif out_value_voltage >= set_value_voltage:
            print("Channel %d is in Voltage Control mode, Out: %.4f, Set: %.4f" % (channel, out_value_voltage, set_value_voltage))
        else:
            print('Output Control Mode is unclear.', out_value_current, set_value_current, out_value_voltage, set_value_voltage)
        return 
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

        print(initial_values)

        # Get values at first from 'initial_values' and overwrite them afterwards with values given in the experiment script
        dtypes = [('v%d' % (i + 1), np.float32) for i in range(4)] + \
                 [('c%d' % (i + 1), np.float32) for i in range(4)]
        
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

        return True


##############################################################################################################
#                                                RunViewer                                                   #
##############################################################################################################

@runviewer_parser
class HP_6622AParser(object):
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












