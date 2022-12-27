from __future__ import division, unicode_literals, print_function, absolute_import
# from labscript_utils import PY2
# if PY2:
#     str = unicode

from labscript_devices import runviewer_parser, BLACS_tab
from labscript import config, Device, IntermediateDevice, StaticAnalogQuantity, LabscriptError, set_passed_properties
import numpy as np
import labscript_utils.h5_lock
import h5py
import socket
import time


ENABLE_RANGE = False

# The Output-Frequency is the AM frequency!!!
MIN_FREQUENCY = 1000  # in Hz
MAX_FREQUENCY = 100000  # in Hz


class Siglent_SSG3000X_SignalGenerator(IntermediateDevice):
    allowed_children = [StaticAnalogQuantity]

    description = 'Siglent synthesized signal generator'

    def __init__(self, name, GPIB_address, **kwargs):
        Device.__init__(self, name, None, 'GPIB', **kwargs)

        self.instructions = {}

        self.GPIB_address = GPIB_address
        self.BLACS_connection = self.GPIB_address

    def add_device(self, output):
        # This device has 2 valid connection-ports: "freqeuncy" and "range"
        # range may be disabled
        for dev in self.child_devices:
            if output.connection == dev.connection:
                # e.g.: only one frequency AnalogQuantitiy is allowed
                raise LabscriptError("Cannot add more than 1 output to {:s} on the same connection {:s}".format(self.name, output.connection))
        if not output.connection in ("frequency", "range", "ref"):
            raise LabscriptError("Cannot attach '{:s}' to '{:s}'. The connection must be a valid value. ({:s})".format(output.name, self.name, "frequency, range, ref"))
        if not ENABLE_RANGE and output.connection == "range":
            raise LabscriptError("Cannot attach '{:s}' to '{:s}' on port 'range'. Range is disabled.".format(output.name, self.name))
        IntermediateDevice.add_device(self, output)

    def generate_code(self, hdf5_file):
        # do checks
        output_frequency = 20000  # 3000000000# in Hz
        output_range = -1
        output_ref = 1    # 1 == 'Int', 2 == 'Ext'

        # iterate through the conneted child devices
        for dev in self.child_devices:
            if dev.connection == 'frequency':
                output_frequency = dev.static_value
            elif dev.connection == 'range':
                output_range = dev.static_value
            elif dev.connection == 'ref':       # int/ ext ref signal source
                output_ref = dev.static_value

        # checks
        if output_frequency < MIN_FREQUENCY:
            raise LabscriptError("The frequency specified for {:s} is smaller than {:d}Hz".format(self.name, MIN_FREQUENCY))
        elif output_frequency > MAX_FREQUENCY:
            raise LabscriptError("The frequency specified for {:s} is larger than {:d}Hz".format(self.name, MAX_FREQUENCY))

        IntermediateDevice.generate_code(self, hdf5_file)

        frequency_table = np.empty(1, dtype={'names': ['frequency', 'range'], 'formats': [np.int64, np.int8]})  # name the column header
        # 1 line with 2 columns
        if not ENABLE_RANGE and output_range != -1:
            output_range = -1
            raise LabscriptError("Cannot set range. Range is disabled")

        frequency_table[0][0] = round(output_frequency)  # get rid of unprecise float numbers
        grp = self.init_device_group(hdf5_file)

        grp.create_dataset('FREQUENCY_OUTPUT', compression=config.compression, data=frequency_table)


import os
from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED

from blacs.device_base_class import DeviceTab

from qtutils import UiLoader
from qtutils.qt.QtCore import *
from qtutils.qt.QtGui import QDoubleValidator


@BLACS_tab
class Siglent_SSG3000X_SignalGeneratorTab(DeviceTab):

    def initialise_GUI(self):
        layout = self.get_tab_layout()
        ui_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'HP_8672A.ui')
        self.ui = UiLoader().load(ui_filepath)
        layout.addWidget(self.ui)

        # Auto ramping setup
        self.auto_ramping = False  # enable auto-ramping by default
        self.ramping_step_size = 0.02  # kHz
        self.ramping_delay = 100  # ms
        self.ramping_target_value = 190.000  # kHz

        # Capabilities
        self.f_base_units = 'kHz'  # front panel values are in MHz!
        self.f_base_min = MIN_FREQUENCY / 1E3  # 2000
        self.f_base_max = MAX_FREQUENCY / 1E3  # 18000
        self.f_base_step = 0.001  # step size for +/- buttons
        self.f_base_decimals = 6  # display 6 decimals accuracy

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

        self.create_analog_outputs(analog_properties)

        _, ao_widgets, _ = self.auto_create_widgets()

        self.auto_place_widgets(ao_widgets)

        self.delay_ramping_counter = 0   # a counter to implement a delay for ramping, to prevent ramping in an state between to shots in an series

        self.ui.cb_auto_ramping.setCheckState(Qt.Checked if self.auto_ramping else Qt.Unchecked)
        self.ui.cb_auto_ramping.stateChanged.connect(self.onAutoRampChanged)

        ramping_validator = QDoubleValidator(MIN_FREQUENCY / 1E3, MAX_FREQUENCY / 1E3, 3)  # MHz
        self.ui.le_auto_ramping.setValidator(ramping_validator)
        self.ui.le_auto_ramping.setText(str(self.ramping_target_value))
        self.ui.le_auto_ramping.textChanged.connect(self.onRampingFrequencyChanged)

        self.ui.le_ramping_step_size.setText(str(self.ramping_step_size))
        self.ui.le_ramping_step_size.textChanged.connect(self.RampingStepSizeChanged)

        self.ui.le_ramping_delay.setText(str(self.ramping_delay))
        self.ui.le_ramping_delay.textChanged.connect(self.RampingDelayChanged)

        self.ui.lbl_ramping_status.setText("")

        connection_table_entry = self.settings['connection_table'].find_by_name(self.settings['device_name'])

        self.IP_address = connection_table_entry.BLACS_connection

        if self.auto_ramping:  # if ramping is enabled, start the 'timer' for auto-ramping
            self.statemachine_timeout_add(self.ramping_delay, self.auto_ramp)  # every tick call auto_ramp()

    def get_save_data(self):
        return {"enable_auto_ramping": self.auto_ramping, "auto_ramping_value": self.ramping_target_value, "auto_ramping_delay": self.ramping_delay, "auto_ramping_step_size": self.ramping_step_size}

    def restore_save_data(self, data):
        if data:  # if its empty, dont restore anything
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
    def transition_to_buffered(self, h5_file, notify_queue):
        DeviceTab.transition_to_buffered(self, h5_file, notify_queue)
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
        worker_initialisation_kwargs = {'GPIB_address': self.GPIB_address}
        self.create_worker("main_worker", RS_SignalGeneratorWorker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"


from labscript_devices.GPIBDevice import GPIBWorker


class RS_SignalGeneratorWorker(GPIBWorker):

    def send_frequency(self, frequency=None, ref=1):
        # update the synthesizer with the given frequency.
        # If an argument is None, the corresponding value will not be changed
        if ref == 2:
            frequency = int(frequency)  # cast frequency to int!
            # frequency in Hz!!
            self.send_string(sendStr)

            if frequency < MIN_FREQUENCY or frequency > MAX_FREQUENCY:
                raise Exception("Frequency {:d} is out of range {:d} - {:d}. Is the frequency in Hz?".format(frequency, MIN_FREQUENCY, MAX_FREQUENCY))

        if frequency is not None:
            sendStr = "AM:FREQ " + str(frequency) + " kHz\r\n"  # Send AM Frequency!
            self.send_string(sendStr)

    def program_manual(self, front_panel_values):
        frequ = front_panel_values['frequency']  # in kHz!
        # refSignal = front_panel_values['ref']  # Int or Ext ref signal
        self.send_frequency(frequ)  # , refSignal)  # still in kHz
        self.send_string('FREQ 90 MHZ')
        self.send_string('OUTP ON')
        self.send_string('AM:STAT ON')
        self.send_string('AM:WAVE SINE')
        self.send_string('AM:DEPT 0.2')
        self.send_string('MOD ON')

        return {}  # no need to adjust the values

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        frequency = initial_values['frequency']
        # refSignal = initial_values['ref']   # Int or Ext ref signal

        with h5py.File(h5file, 'r') as hdf5_file:
            group = hdf5_file['devices/'][device_name]
            data = group.get('FREQUENCY_OUTPUT')
            frequency = data[0][0]  # in Hz

        self.send_frequency(frequency / 1e3)  # , refSignal)

        return {'frequency': frequency / 1e3}

    def send_string(self, string):
        self.socket.sendall(string.encode('utf-8'))
        time.sleep(.300)


@runviewer_parser
class RunviewerClass(object):

    def __init__(self, path, device):
        self.path = path
        self.name = device.name
        self.device = device

    def get_traces(self, add_trace, clock=None):
        # the clock argument is used as stop time, because this device is not connected to any Masterclock
        if not (isinstance(clock, float) or isinstance(clock, int)):  # shoud never happen
            raise Exception("No stop time is passed to RS SignalGenerator")
        else:
            stop_time = clock

        # get the shot data
        with h5py.File(self.path, 'r') as f:
            if 'FREQUENCY_OUTPUT' in f['devices/%s' % self.name]:
                data = f['devices/%s/FREQUENCY_OUTPUT' % self.name][:]
                frequency = data[0][0]
            else:
                frequency = None

        traces = {}
        traces['frequency'] = ((0, stop_time), (frequency, frequency))  # add start- and end point to the trace

        triggers = {}
        for channel_name, channel in self.device.child_list.items():
            if channel.parent_port in traces:
                if channel.device_class == 'Trigger':
                    triggers[channel_name] = traces[channel.parent_port]
                add_trace(channel_name, traces[channel.parent_port], self.name, channel.parent_port)

        return triggers
