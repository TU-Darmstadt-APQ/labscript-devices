"""Hardware-free checks for NI-DAQmx input scaling and live display modes.

Run with QT_QPA_PLATFORM=offscreen python -m unittest
labscript_devices.NI_DAQmx.testing.test_analog_input_display
"""

import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyqtgraph as pg
from qtutils.qt.QtWidgets import QApplication, QWidget, QVBoxLayout
from labscript_utils.qtwidgets.analoginput import AnalogInput
from blacs.device_base_class import DeviceTab

from labscript_devices.NI_DAQmx.analog_input_plot import (
    MAX_DATA, NIAnalogInputPlotWindow,
)
from labscript_devices.NI_DAQmx.blacs_tabs import NI_DAQmxTab


class InputDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.parent = QWidget()
        layout = QVBoxLayout(self.parent)
        self.widgets = {
            'ai%d' % i: AnalogInput('daq', 'ai%d' % i) for i in range(5)
        }
        properties = [
            {'scale_factor': 5, 'units': 'mA'},
            {'scale_factor': -2, 'units': 'A'},
            {'scale_factor': 0, 'units': 'uA'},
            {},
        ]
        devices = {
            'ai%d' % i: SimpleNamespace(properties=props)
            for i, props in enumerate(properties)
        }
        self.tab = NI_DAQmxTab.__new__(NI_DAQmxTab)
        self.tab._device_name = 'daq'
        self.tab.connection_table = SimpleNamespace(
            find_child=lambda name, channel: devices.get(channel)
        )
        with patch.object(DeviceTab, 'create_analog_input_widgets', return_value=self.widgets), \
                patch.object(self.tab, 'get_tab_layout', return_value=layout):
            self.tab.create_analog_input_widgets({})

    def tearDown(self):
        for widget in self.widgets.values():
            widget.close()
        self.parent.close()

    def test_scaling_switch_and_saved_mode(self):
        for widget in self.widgets.values():
            widget.set_value(2)
        for channel, value, units in (
            ('ai0', '10.0000', 'mA'), ('ai1', '-4.0000', 'A'),
            ('ai2', '0.0000', 'uA'), ('ai3', '2.0000', 'V'),
            ('ai4', '2.0000', 'V'),
        ):
            self.assertEqual(self.widgets[channel]._line_edit.text(), value)
            self.assertTrue(self.widgets[channel]._label.text().endswith('(%s)' % units))
        self.tab._ai_show_volts.setChecked(True)
        for widget in self.widgets.values():
            self.assertEqual(widget._line_edit.text(), '2.0000')
            self.assertTrue(widget._label.text().endswith('(V)'))
        saved = self.tab.get_save_data()
        self.tab._ai_show_volts.setChecked(False)
        self.assertEqual(self.widgets['ai0']._line_edit.text(), '10.0000')
        self.tab.restore_save_data(saved)
        self.assertEqual(self.widgets['ai0']._line_edit.text(), '2.0000')
        self.tab.restore_save_data({})
        self.assertEqual(self.widgets['ai0']._line_edit.text(), '10.0000')

    def test_no_value_and_threaded_updates(self):
        self.tab._ai_show_volts.setChecked(True)
        self.tab._ai_show_volts.setChecked(False)
        self.assertEqual(self.widgets['ai0']._line_edit.text(), 'no value')
        thread = threading.Thread(target=self.widgets['ai0'].set_value, args=(2,))
        thread.start()
        deadline = time.monotonic() + 5
        while thread.is_alive() and time.monotonic() < deadline:
            self.app.processEvents()
            thread.join(0.001)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.widgets['ai0']._line_edit.text(), '10.0000')

    def test_plot_open_toggle_focus_and_reopen(self):
        widget = self.widgets['ai0']
        processes = [Mock(), Mock()]
        for process in processes:
            process.start.return_value = Mock(), Mock()
        with patch('labscript_devices.NI_DAQmx.blacs_tabs.NIAnalogInputPlotWindow',
                   side_effect=processes), \
                patch.object(widget, '_check_plot_window'):
            widget._plot_btn.click()
            self.assertEqual(processes[0].start.call_args.args[-3:], (5, 'mA', False))
            self.tab._ai_show_volts.setChecked(True)
            widget.to_child.put.assert_called_with(('show_volts', True))
            widget._plot_btn.click()
            widget.to_child.put.assert_called_with('focus')
            widget.win = None
            widget._plot_btn.click()
            self.assertEqual(processes[1].start.call_args.args[-3:], (5, 'mA', True))


class InputPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_history_is_retained_in_volts(self):
        plot = NIAnalogInputPlotWindow.__new__(NIAnalogInputPlotWindow)
        plot.data = np.array([], dtype=float)
        plot.scale_factor, plot.units, plot.show_volts = 5, 'mA', False
        plot.plot_win = pg.PlotWidget()
        self.addCleanup(plot.plot_win.close)

        def displayed_values():
            self.app.processEvents()
            return plot.plot_win.listDataItems()[0].getData()[1]

        plot.update_plot(np.array([1., 2.]))
        np.testing.assert_array_equal(displayed_values(), [5, 10])
        self.assertEqual(plot.plot_win.getAxis('left').labelUnits, 'mA')
        plot.set_show_volts(True)
        np.testing.assert_array_equal(displayed_values(), [1, 2])
        self.assertEqual(plot.plot_win.getAxis('left').labelUnits, 'V')
        plot.update_plot(np.array([3.]))
        plot.set_show_volts(False)
        np.testing.assert_array_equal(displayed_values(), [5, 10, 15])
        for factor in (0, -2, 5):
            plot.scale_factor = factor
            plot.set_show_volts(False)
            np.testing.assert_array_equal(displayed_values(), np.array([1, 2, 3]) * factor)
        plot.update_plot(np.arange(MAX_DATA + 10, dtype=float))
        self.app.processEvents()
        np.testing.assert_array_equal(plot.data, np.arange(10, MAX_DATA + 10))


class ConnectionTableTests(unittest.TestCase):
    def test_compiled_connection_table_contains_scaling(self):
        script = textwrap.dedent('''
            from labscript import AnalogIn, labscript_init, start, stop
            from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock
            from labscript_devices.NI_DAQmx.models.NI_USB_6343 import NI_USB_6343
            from labscript_utils.connections import ConnectionTable

            labscript_init('connection_table.h5', new=True, overwrite=True)
            clock = DummyPseudoclock('clock')
            daq = NI_USB_6343('daq', clock.clockline, MAX_name='Dev1',
                             clock_terminal='/Dev1/PFI0', acquisition_rate=1000)
            AnalogIn('current', daq, 'ai0', scale_factor=5, units='mA')
            AnalogIn('voltage', daq, 'ai1')
            start()
            stop(0.01)
            table = ConnectionTable('connection_table.h5')
            assert table.find_child('daq', 'ai0').properties['scale_factor'] == 5
            assert table.find_child('daq', 'ai0').properties['units'] == 'mA'
            assert table.find_child('daq', 'ai1').properties['scale_factor'] == 1
            assert table.find_child('daq', 'ai1').properties['units'] == 'Volts'
        ''')
        with tempfile.TemporaryDirectory() as directory:
            script_path = Path(directory) / 'connection_table.py'
            script_path.write_text(script)
            result = subprocess.run([sys.executable, str(script_path)], cwd=directory,
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
