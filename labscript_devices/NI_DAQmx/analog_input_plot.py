"""NI-DAQmx live plots with selectable voltage or scaled input units."""

import threading

import numpy as np
import pyqtgraph as pg
import zmq
from qtutils import inmain_decorator
from qtutils.qt.QtWidgets import QApplication
from labscript_utils.labconfig import LabConfig
from labscript_utils.qtwidgets.InputPlotWindow import MAX_DATA, PlotWindow


class NIAnalogInputPlotWindow(PlotWindow):
    def run(self, connection_name, hardware_name, device_name,
            scale_factor=1.0, units='V', show_volts=False):
        self.scale_factor = scale_factor
        self.units = units
        self.show_volts = show_volts
        self._connection_name = connection_name
        self._hardware_name = hardware_name
        self._device_name = device_name
        self.data = np.array([], dtype=np.float64)
        title = hardware_name
        if connection_name != '-':
            title += ' (%s)' % connection_name
        self.plot_win = pg.plot([], title=title)
        self._redraw()

        broker_port = int(LabConfig().get('ports', 'BLACS_Broker_Pub'))
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect('tcp://127.0.0.1:%d' % broker_port)
        self.socket.setsockopt(
            zmq.SUBSCRIBE, ('%s %s\0' % (device_name, hardware_name)).encode('utf-8')
        )
        threading.Thread(target=self._analog_read_loop, daemon=True).start()
        threading.Thread(target=self._cmd_loop, daemon=True).start()
        QApplication.instance().exec_()
        self.to_parent.put('closed')

    def _cmd_loop(self):
        while True:
            command = self.from_parent.get()
            if command == 'focus':
                self.setTopLevelWindow()
            elif command[0] == 'show_volts':
                self.set_show_volts(command[1])

    @inmain_decorator(False)
    def setTopLevelWindow(self):
        self.plot_win.activateWindow()
        self.plot_win.raise_()

    @inmain_decorator(False)
    def set_show_volts(self, show_volts):
        self.show_volts = show_volts
        self._redraw()

    @inmain_decorator(False)
    def update_plot(self, new_data):
        # Keep history in volts so every mode change can redraw it without loss.
        self.data = np.concatenate((self.data, new_data))[-MAX_DATA:]
        self._redraw()

    def _redraw(self):
        factor = 1.0 if self.show_volts else self.scale_factor
        units = 'V' if self.show_volts else self.units
        self.plot_win.setLabel('left', units=units)
        self.plot_win.plot(self.data * factor, clear=True)
