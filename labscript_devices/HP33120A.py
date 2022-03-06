import sys
from labscript_devices import BLACS_tab, runviewer_parser
from labscript import config, IntermediateDevice, Device, LabscriptError, set_passed_properties
from qtutils.qt.QtWidgets import QWidget, QLineEdit
from labscript_utils.qtwidgets.toolpalette import ToolPaletteGroup
import numpy as np
import pyqtgraph as pg
import threading
MIN_FREQUENCY = 1  # in Hz
MAX_FREQUENCY = 150000000  # in Hz
MIN_VOLTAGE = 0.05  # in V
MAX_VOLTAGE = 10  # in V

MIN_OFFSET = -0.1  # in V
MAX_OFFSET = 0.1  # in V


class HP33120A(IntermediateDevice):
    allowed_children = []

    description = 'HP 33120A AWG'

    def __init__(self, name, GPIB_address, frequency, amplitude, offset, waveform, ext_trigger=False):
        Device.__init__(self, name, None, 'GPIB')

        self.instructions = {}

        self.GPIB_address = GPIB_address
        self.BLACS_connection = self.GPIB_address
        self.frequency = frequency
        self.offset = offset
        self.amplitude = amplitude
        self.ext_trigger = ext_trigger

        if isinstance(waveform, list):
            abs_max = np.max(np.abs(waveform))
            if abs_max > 1:
                self.waveform = waveform / abs_max
            else:
                self.waveform = waveform

    def generate_code(self, hdf5_file):
        # do checks

        IntermediateDevice.generate_code(self, hdf5_file)
        settings_table = np.empty(1, dtype={'names': ['frequency', 'amplitude', 'offset', 'waveform', 'ext_trigger'], 'formats': [np.int64, np.float32, np.float32, np.float32, np.bool_]})  # name the column header

        settings_table[0][0] = self.frequency
        settings_table[0][1] = self.amplitude
        settings_table[0][2] = self.offset
        settings_table[0][3] = self.ext_trigger

        wave_table = np.asarray(self.waveform)

        grp = self.init_device_group(hdf5_file)
        grp.create_dataset('AWG_SETTINGS', compression=config.compression, data=settings_table)
        grp.create_dataset('AWG_WAVE', compression=config.compression, data=wave_table)


from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED

from blacs.device_base_class import DeviceTab


@BLACS_tab
class HP33120A_Tab(DeviceTab):
    def initialise_GUI(self):
        layout = self.get_tab_layout()

        analog_properties = {}

        # Capabilities
        self.f_base_units = 'Hz'  # front panel values are in MHz!
        self.f_base_min = MIN_FREQUENCY  # 2000
        self.f_base_max = MAX_FREQUENCY  # 18000
        self.f_base_step = 1.0  # step size for +/- buttons
        self.f_base_decimals = 0  # display 3 decimals accuracy

        self.a_base_units = 'V'  # front panel values are in MHz!
        self.a_base_min = MIN_VOLTAGE
        self.a_base_max = MAX_VOLTAGE
        self.a_base_step = 0.01  # step size for +/- buttons
        self.a_base_decimals = 2  # display 3 decimals accuracy

        self.offset_base_units = 'V'  # front panel values are in MHz!
        self.offset_base_min = MIN_OFFSET
        self.offset_base_max = MAX_OFFSET
        self.offset_base_step = 0.001  # step size for +/- buttons
        self.offset_base_decimals = 3  # display 3 decimals accuracy

        analog_properties['frequency'] = {'base_unit': self.f_base_units,
                                          'min': self.f_base_min,
                                          'max': self.f_base_max,
                                          'step': self.f_base_step,
                                          'decimals': self.f_base_decimals
                                          }

        analog_properties['amplitude'] = {'base_unit': self.a_base_units,
                                          'min': self.a_base_min,
                                          'max': self.a_base_max,
                                          'step': self.a_base_step,
                                          'decimals': self.a_base_decimals
                                          }

        analog_properties['offset'] = {'base_unit': self.offset_base_units,
                                       'min': self.offset_base_min,
                                       'max': self.offset_base_max,
                                       'step': self.offset_base_step,
                                       'decimals': self.offset_base_decimals
                                       }

        self.create_analog_outputs(analog_properties)

        _, ao_widgets, _ = self.auto_create_widgets()

        widget = QWidget()
        toolpalettegroup = ToolPaletteGroup(widget)
        toolpalette = toolpalettegroup.append_new_palette("Waveform")
        self.waveInput = QLineEdit("[sin(2*pi*x/100) for x in range(100) ]")
        self.waveInput.editingFinished.connect(self.setWaveForm)
        toolpalette.addWidget(self.waveInput, True)
        self.plotWidget = pg.PlotWidget()
        self.get_tab_layout().addWidget(widget)

        self.auto_place_widgets(ao_widgets)
        connection_table_entry = self.settings['connection_table'].find_by_name(self.settings['device_name'])
        self.GPIB_address = connection_table_entry.BLACS_connection
        self.get_tab_layout().addWidget(self.plotWidget)
        self.setWaveForm()

    def restore_save_data(self, save_data):
        if 'waveInput' in save_data:
            self.waveInput.setText(save_data['waveInput'])
            self.setWaveForm()
        DeviceTab.restore_save_data(self, save_data)

    def get_save_data(self):
        save_data = DeviceTab.get_save_data(self)
        save_data['waveInput'] = str(self.waveInput.text())
        return save_data

    @define_state(MODE_MANUAL, True, delete_stale_states=True)
    def program_device(self):
        self.update_plot()
        DeviceTab.program_device(self)

    @define_state(MODE_MANUAL, True)
    def update_plot(self):
        amplitude = self._AO['amplitude'].value
        offset = self._AO['offset'].value
        frequency = self._AO['frequency'].value
        values = self.values
        n_rep = 5
        plotXvalues = np.linspace(0, n_rep / frequency, len(values) * n_rep)
        plotYvalues = [value * amplitude + offset for value in values] * n_rep
        self.plotWidget.plot(x=plotXvalues, y=plotYvalues, clear=True)

    @define_state(MODE_MANUAL, True)
    def setWaveForm(self):
        expression = str(self.waveInput.text())

        sandbox = {}
        exec('from pylab import *', sandbox, sandbox)
        exec('from runmanager.functions import *', sandbox, sandbox)
        values = eval(expression, sandbox)
        if isinstance(values, list):
            abs_max = np.max(np.abs(values))
            if abs_max > 1:
                self.values = values / abs_max
            else:
                self.values = values
            yield(self.queue_work(self._primary_worker, 'setWaveForm', self.values))
            self.update_plot()

    @define_state(MODE_MANUAL, True)
    def transition_to_buffered(self, h5_file, notify_queue):
        DeviceTab.transition_to_buffered(self, h5_file, notify_queue)

    @define_state(MODE_BUFFERED, False)
    def transition_to_manual(self, notify_queue, program=False):
        DeviceTab.transition_to_manual(self, notify_queue, program)

    def initialise_workers(self):
        worker_initialisation_kwargs = {
            'GPIB_address': self.GPIB_address,
            'jump_address': str(self.settings['connection_table'].jump_device_address),
        }
        self.create_worker("main_worker", HP33120A_Worker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"


from labscript_devices.GPIBDevice import GPIBWorker


class HP33120A_Worker(GPIBWorker):

    def init(self):
        GPIBWorker.init(self)
        self.frequency = None
        self.amplitude = None
        self.offset = None
        self.wave = None
        self.ext_trigger = False

    def program_manual(self, front_panel_values):
        frequency = front_panel_values['frequency']
        amplitude = front_panel_values['amplitude']
        offset = front_panel_values['offset']
        self.send_GPIB_settings(frequency, amplitude, offset)

        return {}  # no need to adjust the values

    def send_GPIB_settings(self, frequency=None, amplitude=None, offset=None, ext_trigger=False):
        # update the synthesizer with the given frequency and levelRange.
        # If an argument is None, the corresponding value will not be changed
        if frequency is not None:
            frequency = int(frequency)  # cast frequency to int!
            if frequency != self.frequency:
                if not (frequency >= MIN_FREQUENCY and frequency < MAX_FREQUENCY):
                    raise Exception("Frequency {:d} is out of range {:d} - {:d}. Is the frequency in Hz?".format(frequency, MIN_FREQUENCY, MAX_FREQUENCY))

                self.GPIB_connection.write("FREQ {}".format(frequency))
                self.frequency = frequency

        if amplitude is not None:
            amplitude = float(amplitude)  # cast frequency to int!
            if amplitude != self.amplitude:
                if amplitude < MIN_VOLTAGE or amplitude > MAX_VOLTAGE:
                    raise Exception("Amplitude {:d} is out of range {:d} - {:d}. Is the volatage in V?".format(amplitude, MIN_VOLTAGE, MAX_VOLTAGE))

                self.GPIB_connection.write("VOLT {0:.2f}".format(amplitude))
                self.amplitude = amplitude

        if offset is not None:
            offset = float(offset)  # cast frequency to int!
            if offset != self.offset:
                if offset < MIN_OFFSET or offset > MAX_OFFSET:
                    raise Exception("Offset voltage {:d} is out of range {:d} - {:d}. Is the offset volatage in V?".format(offset, MIN_VOLTAGE, MAX_VOLTAGE))

                self.GPIB_connection.write("VOLT:OFFS {0:.3f}".format(offset))
                self.offset = offset

        ext_trigger = bool(ext_trigger)  # cast frequency to int!
        if ext_trigger != self.ext_trigger:
            if ext_trigger:
                self.GPIB_connection.write("TRIG:SOUR EXT")
            else:
                self.GPIB_connection.write("TRIG:SOUR IMM")
            self.ext_trigger = ext_trigger

    def setWaveForm(self, waveform):
        if self.wave != waveform:
            sendString = "DATA VOLATILE"
            for i, value in enumerate(waveform):
                sendString = "{0}, {1:.3f}".format(sendString, float(value))
            self.GPIB_connection.write(sendString)
            self.GPIB_connection.write("DATA:COPY LABSCRIP")
            self.GPIB_connection.write("FUNC:USER LABSCRIP")
            self.GPIB_connection.write("FUNC:SHAPE USER")
            self.wave = waveform


    def run_experiment(self, device_name):
        self.from_master_socket.recv()
        while True:

            self.to_master_socket.send(str.encode(f"fin {device_name}"))
            msg = self.from_master_socket.recv() # load message

            if msg == b'exit':
                return

            self.to_master_socket.send(str.encode(f"rdy {device_name}"))
            msg = self.from_master_socket.recv() # load message


    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        frequency = initial_values['frequency']
        amplitude = initial_values['amplitude']
        offset = initial_values['offset']

        with h5py.File(h5file, 'r') as hdf5_file:
            group = hdf5_file['devices/'][device_name]
            data = group.get('AWG_SETTINGS')
            frequency = data[0][0]  # in Hz
            amplitude = data[0][1]  # in V
            offset = data[0][2]  # in V
            ext_trigger = data[0][3]  # Bool
            wave = list(group.get('AWG_WAVE'))

        self.send_GPIB_settings(frequency, amplitude, offset, ext_trigger)
        self.setWaveForm(wave)

        # start run thread
        self.run_thread = threading.Thread(target=self.run_experiment, args=(device_name,))
        self.run_thread.start()

        return {'frequency': float(frequency), 'amplitude': float(amplitude), 'offset': float(offset)}


@runviewer_parser
class HP33120A_Parser(object):
    pass
