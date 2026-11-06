from labscript_devices import runviewer_parser, BLACS_tab
from labscript import config, Device, IntermediateDevice, StaticAnalogQuantity, LabscriptError, set_passed_properties
from labscript_utils.shared_drive import path_to_local
import numpy as np
import labscript_utils.h5_lock
import h5py

# The synthesizer is capable to adjust the output amplitude via LevelRange.
# To prevent damage to attached devices, this feature is disabled by default.
# So any command which is sent to the device does not change the setting.
# If it is neccessary to adjust the output amplitude programmatically, you can set ENABLE_RANGE to True.
# This activates the connection terminal 'range' in labscript and in the device tab in BLACS.
ENABLE_RANGE = False

MIN_FREQUENCY = 2000000000  # in Hz
MAX_FREQUENCY = 18000000000  # in Hz


class HP_8672A(IntermediateDevice):
    allowed_children = [StaticAnalogQuantity]

    description = 'HP 8672A synthesized signal generator'

    def __init__(self, name, GPIB_address, igor_address, igor_port, **kwargs):
        Device.__init__(self, name, None, 'GPIB', **kwargs)

        self.instructions = {}

        self.GPIB_address = GPIB_address
        self.BLACS_connection = self.GPIB_address

        self.igor_address = igor_address
        self.igor_port = igor_port

    def add_device(self, output):
        # This device has 2 valid connection-ports: "freqeuncy" and "range"
        # range may be disabled
        for dev in self.child_devices:
            if output.connection == dev.connection:
                # e.g.: only one frequency AnalogQuantitiy is allowed
                raise LabscriptError("Cannot add more than 1 output to {:s} on the same connection {:s}".format(self.name, output.connection))
        if not output.connection in ("frequency", "range"):
            raise LabscriptError("Cannot attach '{:s}' to '{:s}'. The connection must be a valid value. ({:s})".format(output.name, self.name, "frequency, range"))
        if not ENABLE_RANGE and output.connection == "range":
            raise LabscriptError("Cannot attach '{:s}' to '{:s}' on port 'range'. Range is disabled.".format(output.name, self.name))
        IntermediateDevice.add_device(self, output)

    def generate_code(self, hdf5_file):
        # do checks
        output_frequency = 3000000000  # in Hz
        output_range = -1
        # iterate through the conneted child devices
        for dev in self.child_devices:
            if dev.connection == 'frequency':
                output_frequency = dev.static_value
            elif dev.connection == 'range':
                output_range = dev.static_value
        if output_frequency < MIN_FREQUENCY:
            raise LabscriptError("The frequency specified for {:s} is smaller than {:d}Hz".format(self.name, MIN_FREQUENCY))
        elif output_frequency > MAX_FREQUENCY:
            raise LabscriptError("The frequency specified for {:s} is larger than {:d}Hz".format(self.name, MAX_FREQUENCY))

        if output_range != -1 and output_range < -110:
            raise LabscriptError("The range specified for {:s} is smaller than {:d}Hz".format(self.name, -110))
        elif output_range != -1 and output_range > 0:
            raise LabscriptError("The range specified for {:s} is larger than {:d}Hz".format(self.name, 0))

        IntermediateDevice.generate_code(self, hdf5_file)

        frequency_table = np.empty(1, dtype={'names': ['frequency', 'range'], 'formats': [np.int64, np.int8]})  # name the column header
        # 1 line with 2 columns
        if not ENABLE_RANGE and output_range != -1:
            output_range = -1
            raise LabscriptError("Cannot set range. Range is disabled")

        frequency_table[0][0] = round(output_frequency)  # get rid of unprecise float numbers
        frequency_table[0][1] = round(output_range)
        grp = self.init_device_group(hdf5_file)

        grp.create_dataset('FREQUENCY_OUTPUT', compression=config.compression, data=frequency_table)

        self.set_property('igor_address', self.igor_address, location='connection_table_properties')
        self.set_property('igor_port', self.igor_port, location='connection_table_properties')


import os
from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED

from blacs.device_base_class import DeviceTab

from qtutils import UiLoader
import socket
from qtutils.qt.QtCore import *
from qtutils.qt.QtGui import QDoubleValidator


@BLACS_tab
class HP_8672ATab(DeviceTab):

    def initialise_GUI(self):
        layout = self.get_tab_layout()
        ui_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'HP_8672A.ui')
        self.ui = UiLoader().load(ui_filepath)
        layout.addWidget(self.ui)

        # Auto ramping setup
        self.auto_ramping = True  # enable auto-ramping by default
        self.ramping_step_size = 1.0  # MHz
        self.ramping_delay = 500  # ms
        self.ramping_target_value = 3291.000  # MHz

        # Capabilities
        self.f_base_units = 'MHz'  # front panel values are in MHz!
        self.f_base_min = MIN_FREQUENCY / 1E6  # 2000
        self.f_base_max = MAX_FREQUENCY / 1E6  # 18000
        self.f_base_step = 0.001  # step size for +/- buttons
        self.f_base_decimals = 3  # display 3 decimals accuracy

        self.level_base_units = 'dBm'
        self.level_base_min = -110
        self.level_base_max = 0
        self.level_base_step = 10
        self.level_base_decimals = 0

        analog_properties = {}
        analog_properties['frequency'] = {'base_unit': self.f_base_units,
                                          'min': self.f_base_min,
                                          'max': self.f_base_max,
                                          'step': self.f_base_step,
                                          'decimals': self.f_base_decimals
                                          }
        if ENABLE_RANGE:
            analog_properties['range'] = {'base_unit': self.level_base_units,
                                          'min': self.level_base_min,
                                          'max': self.level_base_max,
                                          'step': self.level_base_step,
                                          'decimals': self.level_base_decimals
                                          }

        self.create_analog_outputs(analog_properties)

        _, ao_widgets, _ = self.auto_create_widgets()

        self.auto_place_widgets(ao_widgets)

        self.delay_ramping_counter = 0   # a counter to implement a delay for ramping, to prevent ramping in an state between to shots in an series

        self.send_to_igor = False  # dont send frequencies to Igor by default
        self.ui.cb_send_to_igor.setCheckState(Qt.Checked if self.send_to_igor else Qt.Unchecked)
        self.ui.cb_send_to_igor.stateChanged.connect(self.onSendToIgorChanged)

        self.ui.cb_auto_ramping.setCheckState(Qt.Checked if self.auto_ramping else Qt.Unchecked)
        self.ui.cb_auto_ramping.stateChanged.connect(self.onAutoRampChanged)

        ramping_validator = QDoubleValidator(MIN_FREQUENCY / 1E6, MAX_FREQUENCY / 1E6, 3)  # MHz
        self.ui.le_auto_ramping.setValidator(ramping_validator)
        self.ui.le_auto_ramping.setText(str(self.ramping_target_value))
        self.ui.le_auto_ramping.textChanged.connect(self.onRampingFrequencyChanged)

        self.ui.le_ramping_step_size.setText(str(self.ramping_step_size))
        self.ui.le_ramping_step_size.textChanged.connect(self.RampingStepSizeChanged)

        self.ui.le_ramping_delay.setText(str(self.ramping_delay))
        self.ui.le_ramping_delay.textChanged.connect(self.RampingDelayChanged)

        self.ui.lbl_ramping_status.setText("")

        connection_table_entry = self.settings['connection_table'].find_by_name(self.settings['device_name'])
        self.GPIB_address = connection_table_entry.BLACS_connection
        self.igor_address = connection_table_entry.properties['igor_address']
        self.igor_port = connection_table_entry.properties['igor_port']

        if self.auto_ramping:  # if ramping is enabled, start the 'timer' for auto-ramping
            self.statemachine_timeout_add(self.ramping_delay, self.auto_ramp)  # every tick call auto_ramp()

    def get_save_data(self):
        return {"send_to_igor": self.send_to_igor, "enable_auto_ramping": self.auto_ramping, "auto_ramping_value": self.ramping_target_value, "auto_ramping_delay": self.ramping_delay, "auto_ramping_step_size": self.ramping_step_size}

    def restore_save_data(self, data):
        if data:  # if its empty, dont restore anything
            if "send_to_igor" in data:
                self.send_to_igor = data["send_to_igor"]
                self.ui.cb_send_to_igor.setCheckState(Qt.Checked if self.send_to_igor else Qt.Unchecked)

            if "enable_auto_ramping" in data:
                self.auto_ramping = data["enable_auto_ramping"]
                self.ui.cb_auto_ramping.setCheckState(Qt.Checked if self.auto_ramping else Qt.Unchecked)

            if "auto_ramping_value" in data:
                self.ramping_target_value = data["auto_ramping_value"]
                self.ui.le_auto_ramping.setText(str(self.ramping_target_value))

            if "auto_ramping_delay" in data:
                self.ramping_delay = data["auto_ramping_delay"]
                self.ui.le_ramping_delay.setText(str(self.ramping_delay))

            if "auto_ramping_step_size" in data:
                self.ramping_step_size = data["auto_ramping_step_size"]
                self.ui.le_ramping_step_size.setText(str(self.ramping_step_size))

    @define_state(MODE_MANUAL, True)
    def auto_ramp(self, notify_queue=None):
        if self.auto_ramping:
            self.delay_ramping_counter += 1  # increase the delay counter by 1
            if self.delay_ramping_counter <= 8:
                return  # start auto ramping after 4sec (=8*self.ramping_delay) in IDLE mode

            current_freq = self._AO['frequency'].value  # MHz
            if current_freq > self.ramping_target_value:
                self.ui.lbl_ramping_status.setText("ramping...")
                target_freq = current_freq - self.ramping_step_size
                if target_freq <= self.ramping_target_value:
                    target_freq = self.ramping_target_value
                self._AO['frequency'].set_value(target_freq)
            elif current_freq < self.ramping_target_value:
                self.ui.lbl_ramping_status.setText("ramping...")
                target_freq = current_freq + self.ramping_step_size
                if target_freq >= self.ramping_target_value:
                    target_freq = self.ramping_target_value
                self._AO['frequency'].set_value(target_freq)
            else:
                self.ui.lbl_ramping_status.setText("")

    @define_state(MODE_MANUAL, True)
    def onSendToIgorChanged(self, checked):
        if checked == Qt.Checked:
            self.send_to_igor = True
            # submit the change to the worker
            yield(self.queue_work(self._primary_worker, 'setSendToIgor', self.send_to_igor))
        elif checked == Qt.Unchecked:
            self.send_to_igor = False
            # submit the change to the worker
            yield(self.queue_work(self._primary_worker, 'setSendToIgor', self.send_to_igor))

    @define_state(MODE_MANUAL, True)
    def onAutoRampChanged(self, checked):
        self.ui.lbl_ramping_status.setText("")
        if checked == Qt.Checked:
            self.auto_ramping = True
            # start the ramping 'timer'
            self.statemachine_timeout_add(self.ramping_delay, self.auto_ramp)
            # self.ui.le_auto_ramping.setEnabled(True)
        elif checked == Qt.Unchecked:
            self.auto_ramping = False
            # stop the ramping 'timer'
            self.statemachine_timeout_remove(self.auto_ramp)
            # self.ui.le_auto_ramping.setEnabled(False)

    @define_state(MODE_MANUAL, True)
    def onRampingFrequencyChanged(self, newText):
        self.ramping_target_value = float(newText)

    @define_state(MODE_MANUAL, True)
    def RampingStepSizeChanged(self, newText):
        self.ramping_step_size = float(newText)

    @define_state(MODE_MANUAL, True)
    def RampingDelayChanged(self, newText):
        self.ramping_delay = float(newText)

    @define_state(MODE_MANUAL, True)
    def transition_to_buffered(self, h5_filepath, notify_queue):
        # for remote worker to find correct find path:
        if getattr(self, 'is_remote', False):
            h5_filepath = path_to_local(h5_filepath)
        DeviceTab.transition_to_buffered(self, h5_filepath, notify_queue)
        # stop the auto ramping in buffered mode
        self.statemachine_timeout_remove(self.auto_ramp)

    @define_state(MODE_BUFFERED, False)
    def transition_to_manual(self, notify_queue, program=False):
        DeviceTab.transition_to_manual(self, notify_queue, program)
        # start the auto ramping in manual mode (if enabled)
        if self.auto_ramping:
            self.statemachine_timeout_add(self.ramping_delay, self.auto_ramp)
            self.delay_ramping_counter = 0  # reset ramping delay counter if a shot ends, to prevent back ramping in a shot qequence

    def initialise_workers(self):
        worker_initialisation_kwargs = {'GPIB_address': self.GPIB_address, 'igor_address': self.igor_address, 'igor_port': self.igor_port, 'send_to_igor': self.send_to_igor}
        self.create_worker("main_worker", HP_8672AWorker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"


from labscript_devices.GPIBDevice import GPIBWorker


class HP_8672AWorker(GPIBWorker):

    def init(self):
        GPIBWorker.init(self)
        global socket
        import socket

        self.socket = None
        if self.send_to_igor:  # this variable is passed by the method "initialise_workers" in the device tab class.
            self.init_igor_bridge()

    def init_igor_bridge(self):
        # initialize the TCP connection to Igor
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((self.igor_address, self.igor_port))
            self.socket.settimeout(0.130)  # in s
        except:
            self.socket = None

    def setSendToIgor(self, sendToIgor):
        # this method is called from the Tab-class when the checkbox is (un-)checked
        self.send_to_igor = sendToIgor
        if sendToIgor:
            # connect to igor
            self.init_igor_bridge()
        else:
            # if there is a connection, close it
            if self.socket:
                self.socket.close()
                self.socket = None

    def shutdown(self):
        GPIBWorker.shutdown(self)

        if self.send_to_igor and self.socket:
            self.socket.close()  # close Igor bridge (tcp socket)
            self.socket = None

    def send_GPIB_frequency(self, frequency=None, levelRange=None):
        # update the synthesizer with the given frequency and levelRange.
        # If an argument is None, the corresponding value will not be changed
        if frequency is not None:
            frequency = int(frequency)  # cast frequency to int!
            # frequency in Hz!!

            if frequency < MIN_FREQUENCY or frequency > MAX_FREQUENCY:
                raise Exception("Frequency {:d} is out of range {:d} - {:d}. Is the frequency in Hz?".format(frequency, MIN_FREQUENCY, MAX_FREQUENCY))

        if levelRange is not None and not ENABLE_RANGE:
            raise Exception("Cannot update levelRange. This feature is disabled!")

        # get the correct command value for levelRange
        if levelRange is None:
            level_range_index = None
        elif levelRange == 0:
            level_range_index = "0"
        elif levelRange == -10:
            level_range_index = "1"
        elif levelRange == -20:
            level_range_index = "2"
        elif levelRange == -30:
            level_range_index = "3"
        elif levelRange == -40:
            level_range_index = "4"
        elif levelRange == -50:
            level_range_index = "5"
        elif levelRange == -60:
            level_range_index = "6"
        elif levelRange == -70:
            level_range_index = "7"
        elif levelRange == -80:
            level_range_index = "8"
        elif levelRange == -90:
            level_range_index = "9"
        elif levelRange == -100:
            level_range_index = ":"
        elif levelRange == -110:
            level_range_index = ";"
        else:
            raise Exception("Level Range is not a valid value.")

        sendStr = ""
        if frequency is not None:
            # P first digit is  10GHz
            # Q first digit is   1GHz
            # R first digit is 100MHz
            # S first digit is  10MHz
            # T first digit is   1MHz
            # U first digit is 100kHz
            # V first digit is  10kHz
            # W first digit is   1kHz
            sendStr += "P{:08d}".format(int(frequency / 1E3))  # first digit is 10GHz, so print leading zeros!

        if level_range_index is not None:
            # output level range: "K"+[0,1,2,3,4,5,6,7,8,9,:,;] for 0dBm,-10dBm...-110dBm , 10dBm stepsize
            sendStr += "K{:s}".format(level_range_index)

        # send a message only if there is anything to send
        if len(sendStr) > 0:
            # Z0: Execute
            # N6: FM off
            sendStr += "Z0N6"
            self.GPIB_connection.write(sendStr)

    def send_igor_frequency(self, frequency):
        return
        if self.socket is None:
            self.init_igor_bridge()  # if we are not connected, try to connect

        # this method sends the frequency to Igor
        # The send strings are taken from the current LabView program "SigGen.vi"
        self.socket.send("LF_setNumGenerators(1)\n")
        self.socket.send("LF_setFrequencies(0,\"{:.6f};\")\n".format(frequency / 1E3))  # convert Hz to kHz
        self.socket.send("Wait\r\n")

        try:  # wait for/read a reply... (LabView does it)
            reply = self.socket.recv(10)  # read 10bytes reply
            # actually I have not received any reply by now :-(
        except:
            pass

    def program_manual(self, front_panel_values):
        frequ = front_panel_values['frequency']  # in MHz!
        if ENABLE_RANGE:
            level_range = front_panel_values['range']  # in dBm
        else:
            level_range = None

        self.send_GPIB_frequency(frequ * 1E6, level_range)  # convert MHz to Hz

        if self.send_to_igor:
            self.send_igor_frequency(frequ * 1E6)  # convert MHz to Hz

        return {}  # no need to adjust the values

    def transition_to_buffered(self, device_name, h5_filepath, initial_values, fresh):
        # for remote worker to find correct find path:
        if getattr(self, 'is_remote', False):
            h5_filepath = path_to_local(h5_filepath)
        frequency = initial_values['frequency'] * 1e6  # convert to Hz
        if ENABLE_RANGE:
            level_range = initial_values['range']
        else:
            level_range = None

        with h5py.File(h5_filepath, 'r') as hdf5_file:
            group = hdf5_file['devices/'][device_name]
            data = group.get('FREQUENCY_OUTPUT')
            frequency = data[0][0]  # in Hz
            level_range = data[0][1]  # in dBm
            if level_range == -1 or not ENABLE_RANGE:
                level_range = None

        self.send_GPIB_frequency(frequency, level_range)
        if self.send_to_igor:
            self.send_igor_frequency(frequency)

        if level_range:
            return {'frequency': frequency / 1e6, 'range': level_range}  # final values, in MHz
        else:
            return {'frequency': frequency / 1e6}


@runviewer_parser
class RunviewerClass(object):

    def __init__(self, path, device):
        self.path = path
        self.name = device.name
        self.device = device

    def get_traces(self, add_trace, clock=None):
        # the clock argument is used as stop time, because this device is not connected to any Masterclock
        if not (isinstance(clock, float) or isinstance(clock, int)):  # shoud never happen
            raise Exception("No stop time is passed to HP 8672A")
        else:
            stop_time = clock

        # get the shot data
        with h5py.File(self.path, 'r') as f:
            if 'FREQUENCY_OUTPUT' in f['devices/%s' % self.name]:
                data = f['devices/%s/FREQUENCY_OUTPUT' % self.name][:]
                frequency = data[0][0]
                level_range = data[0][1]
            else:
                frequency = None
                level_range = None

        traces = {}
        traces['frequency'] = ((0, stop_time), (frequency, frequency))  # add start- and end point to the trace
        if ENABLE_RANGE and level_range != -1:
            traces['range'] = ((0, stop_time), (level_range, level_range))

        triggers = {}
        for channel_name, channel in self.device.child_list.items():
            if channel.parent_port in traces:
                if channel.device_class == 'Trigger':
                    triggers[channel_name] = traces[channel.parent_port]
                add_trace(channel_name, traces[channel.parent_port], self.name, channel.parent_port)

        return triggers
