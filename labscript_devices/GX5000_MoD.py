from labscript_devices import runviewer_parser, BLACS_tab
from labscript import config, Device, TriggerableDevice, LabscriptError, set_passed_properties
import numpy as np
import labscript_utils.h5_lock
import h5py


class GX5000_MoD(TriggerableDevice):
    allowed_children = []

    description = 'GX5000 MoD Programmable Pulse Generator'

    @set_passed_properties(
        property_names={
            "connection_table_properties": ["GPIB_address"],
            "device_properties": ["trigger_source", "trigger_mode", "pulse_periode", "pulse_width", "pulse_delay", "pulse_polarity"]}
    )
    def __init__(self, name, parent_device, connection, GPIB_address, trigger_source="EXT", trigger_mode="GATE", pulse_periode=0, pulse_width=0, pulse_delay=0, pulse_polarity="NORM", **kwargs):
        if trigger_mode not in ['GATE', 'TRIG', 'BURST', 'CONT']:
            raise Exception('Unknown trigger mode for GX5000. Allowed trigger modes are: CONTinuous, TRIGger, GATE, BURSt')

        if trigger_source not in ['MAN', 'BUS', 'EXT', 'INT']:
            raise Exception('Unknown trigger source for GX5000. Allowed trigger sources are: MANual, BUS, INTernal, EXTernal')

        if pulse_polarity not in ['NORM', 'COMPL', 'INV']:
            raise Exception('Unknown pulse polarity for GX5000. Allowed trigger polarities are: NORMal, COMPLement, INVerted')

        pulse_periode = float(pulse_periode)
        if not (20e-9 < pulse_periode and pulse_periode < 10):
            raise Exception('pulse_periode must be given in seconds. Values need to be greater than 20 ns and smaller than 10 s.')

        pulse_delay = float(pulse_delay)
        # if 0 < pulse_delay < pulse_periode:
        #    raise Exception('pulse_delay must be given in seconds. Values need to be greater than 20 ns and smaller than 10 s.')

        pulse_Width = float(pulse_width)
        if not (10e-9 < pulse_width and pulse_width + pulse_delay < pulse_periode):
            raise Exception('pulse_width must be given in seconds. Values need to be greater than 10 ns and smaller than the pulse_periode.')

        self.trigger_mode = trigger_mode
        self.trigger_source = trigger_source
        self.pulse_polarity = pulse_polarity
        self.pulse_periode = pulse_periode
        self.pulse_delay = pulse_delay
        self.pulse_width = pulse_width

        self.instructions = {}

        self.GPIB_address = GPIB_address
        self.BLACS_connection = GPIB_address

        TriggerableDevice.__init__(self, name, parent_device, connection, **kwargs)

    def generate_code(self, hdf5_file):
        pass


import os
from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED

from blacs.device_base_class import DeviceTab

from qtutils import UiLoader
from qtutils.qt.QtCore import *
from qtutils.qt.QtWidgets import QComboBox, QWidget, QLabel, QGridLayout, QSpacerItem, QHBoxLayout
from qtutils.qt.QtGui import QSizePolicy


@BLACS_tab
class GX5000_MoDTab(DeviceTab):
    def initialise_GUI(self):
        layout = self.get_tab_layout()

        self.GPIB_address = self.settings['connection_table'].find_by_name(self.settings["device_name"]).BLACS_connection

        # Capabilities
        self.f_base_units = 's'  # front panel values are in MHz!
        self.f_base_step = 0.00000001  # step size for +/- buttons
        self.f_base_decimals = 9  # display 9 decimals accuracy
        analog_properties = {}
        analog_properties['Pulse Periode'] = {'base_unit': self.f_base_units,
                                              'min': 20e-9,
                                              'max': 10,
                                              'step': self.f_base_step,
                                              'decimals': self.f_base_decimals
                                              }
        analog_properties['Pulse Width'] = {'base_unit': self.f_base_units,
                                            'min': 10e-9,
                                            'max': 10,
                                            'step': self.f_base_step,
                                            'decimals': self.f_base_decimals
                                            }
        analog_properties['Pulse Delay'] = {'base_unit': self.f_base_units,
                                            'min': 0,
                                            'max': 9.85,
                                            'step': self.f_base_step,
                                            'decimals': self.f_base_decimals
                                            }

        def ao_sort(channel):
            if channel == 'Pulse Periode':
                return 1
            elif channel == 'Pulse Periode':
                return 2
            elif channel == 'Pulse Delay':
                return 3
            elif channel == 'Polarity':
                return 4

        self.trigger_mode = None
        self.trigger_source = None
        self.pulse_polarity = None

        self.create_analog_outputs(analog_properties)

        digital_properties = {}
        digital_properties['Active'] = {}
        self.create_digital_outputs(digital_properties)

        _, pulse_widgets, output_widgets = self.auto_create_widgets()

        def create_widget(label, item):
            widget = QWidget()
            _label = QLabel(label)
            _label.setAlignment(Qt.AlignCenter)
            _label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
            widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
            item.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
            item.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            _layout = QGridLayout(widget)
            _layout.setVerticalSpacing(0)
            _layout.setHorizontalSpacing(0)
            _layout.setContentsMargins(5, 5, 5, 5)
            _label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
            _layout.addWidget(_label)
            h_widget = QWidget()
            h_layout = QHBoxLayout(h_widget)
            h_layout.setContentsMargins(0, 0, 0, 0)
            h_layout.addWidget(item)
            _layout.addWidget(_label, 0, 0)
            _layout.addWidget(h_widget, 1, 0)

            return widget

        self.combobox_Polarity = QComboBox()
        self.combobox_Polarity.addItems(['NORMal', 'COMPLement', 'INVerted'])
        self.combobox_Polarity.currentTextChanged.connect(lambda x: self.program_device())
        pulse_widgets['Polarity'] = create_widget("Polarity", self.combobox_Polarity)

        trigger_widgets = {}
        self.combobox_TrigMode = QComboBox()
        self.combobox_TrigMode.addItems(['CONTinuous', 'TRIGger', 'GATE', 'BURSt'])
        self.combobox_TrigMode.currentTextChanged.connect(lambda x: self.program_device())
        trigger_widgets['Trigger Mode'] = create_widget("Trigger Mode", self.combobox_TrigMode)

        self.combobox_TrigSrc = QComboBox()
        self.combobox_TrigSrc.addItems(['MANual', 'BUS', 'INTernal', 'EXTernal'])
        self.combobox_TrigSrc.currentTextChanged.connect(lambda x: self.program_device())
        trigger_widgets['Trigger Source'] = create_widget("Trigger Source", self.combobox_TrigSrc)

        self.auto_place_widgets(("Pulse Settings", pulse_widgets, ao_sort), ("Trigger Settings", trigger_widgets), ("Outputs Settings", output_widgets))

    def get_front_panel_values(self):
        front_panel_values = DeviceTab.get_front_panel_values(self)
        front_panel_values['Polarity'] = self.combobox_Polarity.currentText()
        front_panel_values['Trigger Mode'] = self.combobox_TrigMode.currentText()
        front_panel_values['Trigger Source'] = self.combobox_TrigSrc.currentText()
        return front_panel_values

    def get_save_data(self):
        return {'Polarity': self.combobox_Polarity.currentText(), 'Trigger Mode': self.combobox_TrigMode.currentText(), 'Trigger Source': self.combobox_TrigSrc.currentText()}

    def restore_save_data(self, data):
        if data:  # if its empty, dont restore anything
            if 'Polarity' in data:
                self.combobox_Polarity.setCurrentIndex(self.combobox_Polarity.findText(data['Polarity']))
            if 'Trigger Mode' in data:
                self.combobox_TrigMode.setCurrentIndex(self.combobox_TrigMode.findText(data['Trigger Mode']))
            if 'Trigger Source' in data:
                self.combobox_TrigSrc.setCurrentIndex(self.combobox_TrigSrc.findText(data['Trigger Source']))

    @define_state(MODE_MANUAL, True)
    def transition_to_buffered(self, h5_file, notify_queue):
        DeviceTab.transition_to_buffered(self, h5_file, notify_queue)

    @define_state(MODE_BUFFERED, False)
    def transition_to_manual(self, notify_queue, program=False):
        DeviceTab.transition_to_manual(self, notify_queue, program)

        values_dict = {'NORM': 'NORMal', 'COMPL': 'COMPLement', 'INV': 'INVerted'}
        if self._final_values['Polarity'] in values_dict:
            self._final_values['Polarity'] = values_dict[self._final_values['Polarity']]
        self.combobox_Polarity.setCurrentIndex(self.combobox_Polarity.findText(self._final_values['Polarity']))

        values_dict = {'CONT': 'CONTinuous', 'TRIG': 'TRIGger', 'GATE': 'GATE', 'BURS': 'BURSt'}
        if self._final_values['Trigger Mode'] in values_dict:
            self._final_values['Trigger Mode'] = values_dict[self._final_values['Trigger Mode']]
        self.combobox_TrigMode.setCurrentIndex(self.combobox_TrigMode.findText(self._final_values['Trigger Mode']))

        values_dict = {'MAN': 'MANual', 'INT': 'INTernal', 'EXT': 'EXTernal'}
        if self._final_values['Trigger Source'] in values_dict:
            self._final_values['Trigger Source'] = values_dict[self._final_values['Trigger Source']]
        self.combobox_TrigSrc.setCurrentIndex(self.combobox_TrigSrc.findText(self._final_values['Trigger Source']))

    def initialise_workers(self):
        worker_initialisation_kwargs = {'GPIB_address': self.GPIB_address}
        self.create_worker("main_worker", GX5000_MoDWorker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"


from labscript_devices.GPIBDevice import GPIBWorker


class GX5000_MoDWorker(GPIBWorker):

    def init(self):
        GPIBWorker.init(self)
        self.trigger_mode = None
        self.trigger_source = None
        self.pulse_polarity = None
        self.pulse_periode = None
        self.pulse_delay = None
        self.pulse_width = None
        self.output = None

    def program_manual(self, front_panel_values):
        self.programm_GPIB(trigger_source=front_panel_values['Trigger Source'], trigger_mode=front_panel_values['Trigger Mode'], pulse_periode=front_panel_values['Pulse Periode'],
                           pulse_width=front_panel_values['Pulse Width'], pulse_delay=front_panel_values['Pulse Delay'], pulse_polarity=front_panel_values['Polarity'], output="ON" if front_panel_values['Active'] else "OFF")

    def programm_GPIB(self, trigger_source="EXT", trigger_mode="GATE", pulse_periode=0, pulse_width=0, pulse_delay=0, pulse_polarity="NORM", output="ON", no_check=False):

        trigger_source = filter(lambda x: x.isupper(), trigger_source)
        trigger_mode = filter(lambda x: x.isupper(), trigger_mode)
        pulse_polarity = filter(lambda x: x.isupper(), pulse_polarity)
        output = filter(lambda x: x.isupper(), output)

        if self.pulse_periode != pulse_periode or no_check:
            self.GPIB_connection.write(":SOUR:PULS:PER {}".format(self.time_convert(pulse_periode)).rstrip('\n\r '))
            self.pulse_periode = pulse_periode

        if self.pulse_delay != pulse_delay or no_check:
            self.GPIB_connection.write(":SOUR:PULS:DEL {}".format(self.time_convert(pulse_delay)))
            self.pulse_delay = pulse_delay

        if self.pulse_width != pulse_width or no_check:
            self.GPIB_connection.write(":SOUR:PULS:WIDT {}".format(self.time_convert(pulse_width)))
            self.pulse_width = pulse_width

        if self.trigger_mode != trigger_mode or no_check:
            self.GPIB_connection.write(":TRIG:MODE {}".format(trigger_mode))
            self.trigger_mode = trigger_mode

        if self.trigger_source != trigger_source or no_check:
            self.GPIB_connection.write(":TRIG:SOUR {}".format(trigger_source))
            self.trigger_source = trigger_source

        if self.pulse_polarity != pulse_polarity or no_check:
            self.GPIB_connection.write(":SOUR:PULS:POL {}".format(pulse_polarity))
            self.pulse_polarity = pulse_polarity

        if self.output != output or no_check:
            self.GPIB_connection.write(":OUTP:STAT {}".format(output))
            self.output = output

    def time_convert(self, time):
        number = time * 1e9
        unit = "NS"
        return "{}{}".format(int(number), unit)

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        with h5py.File(h5file, 'r') as hdf5_file:
            device_properties = labscript_utils.properties.get(hdf5_file, device_name, 'device_properties')
        del device_properties['added_properties']
        self.programm_GPIB(**device_properties)

        final_values = {}
        final_values["Trigger Source"] = device_properties["trigger_source"]
        final_values["Trigger Mode"] = device_properties["trigger_mode"]
        final_values["Pulse Periode"] = device_properties["pulse_periode"]
        final_values["Pulse Width"] = device_properties["pulse_width"]
        final_values["Pulse Delay"] = device_properties["pulse_delay"]
        final_values["Polarity"] = device_properties["pulse_polarity"]
        final_values["Active"] = False
        return final_values

    def transition_to_manual(self):
        self.programm_GPIB(trigger_source=self.trigger_source, trigger_mode=self.trigger_mode, pulse_periode=self.pulse_periode, pulse_width=self.pulse_width, pulse_delay=self.pulse_delay, pulse_polarity=self.pulse_polarity, output="OFF")
        return True  # return success


@runviewer_parser
class RunviewerClass(object):

    def __init__(self, path, device):
        self.path = path
        self.name = device.name
        self.device = device

    def get_traces(self, add_trace, clock=None):
        return {}
