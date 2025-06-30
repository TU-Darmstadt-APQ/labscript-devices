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

from PyQt5.QtWidgets import QLabel,QWidget,QHBoxLayout
from PyQt5.QtCore import Qt
# from qtutils import UiLoader
# from qtutils.qt.QtCore import *
# from qtutils.qt.QtGui import QDoubleValidator

# --- Worker Imports
from labscript_devices.GPIBDevice import GPIBWorker

# --- Others
from .logger_config import logger



# TODO To investigate It seems that quering the readback values ist happening before setting value
     # which is weid because the order in the code should be otherwise
     # Therfore the readback is currently showing the last set output value

##############################################################################################################
#                                               Device Limits                                                #
##############################################################################################################

# Specifications for HP6626A:
max_no_of_outputs : int = 4
Watt_ratings : list[int] = [25, 25, 50, 50]
voltage_decimals :int = 2
current_decimals :int = 3

# DC Output Range Specifications
LOW_RANGE :bool = False
MIN_VOLTAGE = 0
MAX_VOLTAGE = 16
MIN_CURRENT = 0
MAX_CURRENT = 1

MIN_VOLTAGE_25W_LOW_RANGE = 0  # in V; Outputs 1 and 2
MAX_VOLTAGE_25W_LOW_RANGE = 7  # in V; Outputs 1 and 2
MIN_VOLTAGE_50W_LOW_RANGE = 0  # in V; Outputs 3 and 4
MAX_VOLTAGE_50W_LOW_RANGE = 16  # in V; Outputs 3 and 4

MIN_CURRENT_25W_LOW_RANGE = 0  # in A; Outputs 1 and 2
MAX_CURRENT_25W_LOW_RANGE = 0.015  # in A; Outputs 1 and 2
MIN_CURRENT_50W_LOW_RANGE = 0  # in A; Outputs 3 and 4
MAX_CURRENT_50W_LOW_RANGE = 0.2  # in A; Outputs 3 and 4


MIN_VOLTAGE_25W_HIGH_RANGE = 0  # in V; Outputs 1 and 2
MAX_VOLTAGE_25W_HIGH_RANGE = 50  # in V; Outputs 1 and 2
MIN_VOLTAGE_50W_HIGH_RANGE = 0  # in V; Outputs 3 and 4
MAX_VOLTAGE_50W_HIGH_RANGE = 16  # in V; Outputs 3 and 4

MIN_CURRENT_25W_HIGH_RANGE = 0  # in A; Outputs 1 and 2
MAX_CURRENT_25W_HIGH_RANGE = 0.5  # in A; Outputs 1 and 2
MIN_CURRENT_50W_HIGH_RANGE = 0  # in A; Outputs 3 and 4
MAX_CURRENT_50W_HIGH_RANGE = 2  # in A; Outputs 3 and 4


##############################################################################################################
#                                               DEVICE                                                       #
##############################################################################################################

class HP_6626A(IntermediateDevice):
    '''
        This Devices allows 8 "StaticAnalogQuantity"s to be set: Voltage and Current for each of the four outputs.
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

    description = 'HP 6626A DC Power Supply'

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


    def generate_code(self, hdf5_file):
        IntermediateDevice.generate_code(self, hdf5_file)

        dtypes = [('v%d' % (i + 1), np.float32) for i in range(4)] + \
                 [('c%d' % (i + 1), np.float32) for i in range(4)]
        output_table = np.zeros(1, dtype=dtypes)

        # iterate through the connected child devices
        # Child devices are the StaticAnalogQuantities connected

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
            if output_table['v%d' % i] < MIN_VOLTAGE_25W_HIGH_RANGE or output_table['v%d' % i] > MAX_VOLTAGE_25W_HIGH_RANGE:
                raise LabscriptError("The voltage specified for {:s} is not within the power supply's voltage range".format(device.connection))
        for i in range(2):
            i += 1
            if output_table['c%d' % i] < MIN_CURRENT_25W_HIGH_RANGE or output_table['c%d' % i] > MAX_CURRENT_25W_HIGH_RANGE:
                raise LabscriptError("The voltage specified for {:s} is not within the power supply's voltage range".format(device.connection))
        for i in range(2, 4):
            i += 1
            if output_table['v%d' % i] < MIN_VOLTAGE_50W_HIGH_RANGE or output_table['v%d' % i] > MAX_VOLTAGE_50W_HIGH_RANGE:
                raise LabscriptError("The voltage specified for {:s} is not within the power supply's voltage range".format(device.connection))
        for i in range(2, 4):
            i += 1
            if output_table['c%d' % i] < MIN_CURRENT_50W_HIGH_RANGE or output_table['c%d' % i] > MAX_CURRENT_50W_HIGH_RANGE:
                raise LabscriptError("The voltage specified for {:s} is not within the power supply's voltage range".format(device.connection))

        # print('output_table',output_table)

        # Create device group in the HDF5 file:
        grp = self.init_device_group(hdf5_file)

        # Save Output to HDF5File:
        grp.create_dataset('OUTPUT_DATA', compression=config.compression, data=output_table)




##############################################################################################################
#                                                TAB                                                         #
##############################################################################################################

@BLACS_tab
class HP_6626ATab(DeviceTab):

    def initialise_GUI(self):
        # --- This allows to register a function to be called every periode of time t (ms)
        time_check_state = 5        # in seconds
        time_check_state = time_check_state * 1e3 # conversion to ms
        self.statemachine_timeout_add(time_check_state ,self.status_monitor)

        # --- Connection table properties
        connection_table = self.settings['connection_table']
        connection_table_entry = self.settings['connection_table'].find_by_name(self.settings['device_name'])
        connection_table_properties = connection_table.find_by_name(self.device_name).properties
        self.num_outputs = 4
        self.GPIB_address = connection_table_entry.BLACS_connection

        # --- Capabilities:
        self.base_units = {'v': 'V', 'c': 'A'}
        self.base_step = {'v': 0.1, 'c': 0.01}  # step size for +/- buttons
        self.base_decimals = {'v': voltage_decimals, 'c': current_decimals}  # display 2 decimals accuracy

        analog_properties = {}
        for i in range(2):
            analog_properties['out%d/voltage' % (i + 1)] = {'base_unit': self.base_units['v'],
                                                            'min': MIN_VOLTAGE_25W_HIGH_RANGE,
                                                            'max': MAX_VOLTAGE_25W_HIGH_RANGE,
                                                            'step': self.base_step['v'],
                                                            'decimals': self.base_decimals['v']
                                                            }
            analog_properties['out%d/current' % (i + 1)] = {'base_unit': self.base_units['c'],
                                                            'min': MIN_CURRENT_25W_HIGH_RANGE,
                                                            'max': MAX_CURRENT_25W_HIGH_RANGE,
                                                            'step': self.base_step['c'],
                                                            'decimals': self.base_decimals['c']
                                                            }
        for i in range(2, 4):
            analog_properties['out%d/voltage' % (i + 1)] = {'base_unit': self.base_units['v'],
                                                            'min': MIN_VOLTAGE_50W_HIGH_RANGE,
                                                            'max': MAX_VOLTAGE_50W_HIGH_RANGE,
                                                            'step': self.base_step['v'],
                                                            'decimals': self.base_decimals['v']
                                                            }
            analog_properties['out%d/current' % (i + 1)] = {'base_unit': self.base_units['c'],
                                                            'min': MIN_CURRENT_50W_HIGH_RANGE,
                                                            'max': MAX_CURRENT_50W_HIGH_RANGE,
                                                            'step': self.base_step['c'],
                                                            'decimals': self.base_decimals['c']
                                                            }

        # -------------------------------------------------- Widgets
        # --- Auto widgets
        self.create_analog_outputs(analog_properties)
        _, ao_widgets, _ = self.auto_create_widgets()

        # --- Readback widgets
        int_val = 0
        self.readback_widgets = {}
        for key,widget in ao_widgets.items():
            widget_layout = widget.layout()     # which is a GridLayout Btw
            readback_widget = QLabel(f"Readback: {int_val}")
            self.readback_widgets[key] = readback_widget
            widget_layout.addWidget(readback_widget,2,0,alignment=Qt.AlignmentFlag.AlignTop)   # row # column

        self.auto_place_widgets(ao_widgets)

        # --- Create status a status label
        self.status_labels_dict = {} 
        layout = self.get_tab_layout()
        layout.setSpacing(5) 
        layout.setContentsMargins(5, 5, 5, 5)
        for i in range(1,self.num_outputs +1):
            self.status_label = QLabel(f"Mode channel {i} : Unknown")
            self.status_labels_dict[i] = self.status_label
            layout.addWidget(self.status_label,alignment=Qt.AlignmentFlag.AlignTop)



        # --- Another thing that can be done, but not adequate here
        # self.supports_remote_value_check(True) # This one checks the precise values and asks the user which one to choose

    @define_state(MODE_MANUAL,True,delete_stale_states=True)
    def program_device(self):
        DeviceTab.program_device(self)    # to don't disturb the basic functionalities
        current_output_values = yield(self.queue_work(self.primary_worker,'get_readbacks'))
        logger.info(current_output_values)
        for key, value in current_output_values.items():
            self.readback_widgets[key].setText(f"Readback: {value}")

    @define_state(MODE_MANUAL, True)
    def status_monitor(self):
        for chan in range(1,self.num_outputs +1):
            status_label = self.status_labels_dict[chan] 
            mode = yield (self.queue_work(self.primary_worker, "check_status",chan))
            status_label.setText(f"Mode channel {chan} : {mode}")


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
        self.create_worker("main_worker", HP_6626AWorker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"


##############################################################################################################
#                                                WORKER                                                      #
##############################################################################################################

class HP_6626AWorker(GPIBWorker):

    # -------------------------- Instrument specific methodes
    def set_v(self,chan,voltage):
        try:
            sendStr = f"VSET {chan},{voltage}"
            self.GPIB_connection.write(sendStr)
        except Exception as e:
            raise LabscriptError("Failed setting voltage : ",e)

    def get_v(self,chan):
        # Accuracy of the readback value over an interface is the same as the analog-to-digital converter
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
        # Accuracy of the readback value over an interface is the same as the analog-to-digital converter
        try:
            sendStr = f"IOUT? {chan}"
            return self.GPIB_connection.query(sendStr)
        except Exception as e:
            raise LabscriptError(f"Failed getting current: {e}")
        
    def get_mode(self,chan):
        try:
            sendStr = f"STS? {chan}"
            result = self.GPIB_connection.query(sendStr)
            return int(result)
        except Exception as e:
            raise LabscriptError(f"Failed getting mode: {e}")
        
    # -------------------------- Worker/Instrument methodes
    def send_GPIB_voltage(self, voltage=None, output=None):
        # Update the power supply outputs with the specified voltages.
        # If an argument is None, the corresponding value will not be changed
        if voltage is not None:
            voltage = np.round(voltage, voltage_decimals)  # round voltage to four decimal places!

            if voltage < MIN_VOLTAGE or voltage > MAX_VOLTAGE:
                raise Exception("Voltage {:f} is out of range {:f} to {:f}. Is the voltage in V?".format(voltage, MIN_VOLTAGE, MAX_VOLTAGE))

        if voltage is not None and output is not None:
            self.set_v(output,voltage)

    def send_GPIB_current(self, current=None, output=None):
        # Update the power supply  current outputs with the specified currents.
        # If an argument is None, the corresponding value will not be changed
        if current is not None:
            current = np.round(current, current_decimals)  # round current to three decimal places!

            if current < MIN_CURRENT or current > MAX_CURRENT:
                raise Exception("Current {:f} is out of range {:f} to {:f}. Is the Current in A?".format(current, MIN_CURRENT, MAX_CURRENT))

        if current is not None and output is not None:
            self.set_i(output,current)

    # -------------------------- Worker/Tab methodes
    def check_status(self,chan):
        mode = self.get_mode(chan)
        
        meanings = ['CV', '+CC', '-CC', 'OV', 'OT', 'UNR', 'OC', 'CP']  # Page 74 OPERATING MANUAL 6622A
        status = []

        for i in range(8):
            if mode & (1 << i):  # bitmask where a single bit is set to 1 at the position we're currently checking,
                                       # and all other bits are 0 (left shift operator <<)
                status.append(meanings[i])
        return ' '.join(status)

    def get_readbacks(self):
        current_output_values = {}
        for i in range(1, self.num_outputs + 1 ):
            current_output_values[f'out{i}/voltage'] = np.round(float(self.get_v( i)) , voltage_decimals)
            current_output_values[f'out{i}/current'] = np.round(float(self.get_i( i)), current_decimals)
        return current_output_values

    # -------------------------- The usuals
    def program_manual(self, front_panel_values):
        # Get values from the front_panel_settings
        for i in range(self.num_outputs):
            voltage = front_panel_values['out' + str(i + 1) + '/voltage']
            self.send_GPIB_voltage(voltage=voltage, output=i + 1)

        for i in range(self.num_outputs):
            current = front_panel_values['out' + str(i + 1) + '/current']
            self.send_GPIB_current(current=current, output=i + 1)

        return {}  

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
class HP_6626AParser(object):
    pass
