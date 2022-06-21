from labscript_devices import runviewer_parser, BLACS_tab
from labscript import config, Device, IntermediateDevice, StaticAnalogQuantity, LabscriptError, set_passed_properties
import numpy as np
import labscript_utils.h5_lock
import h5py


class EIP_545A(IntermediateDevice):
    allowed_children = [StaticAnalogQuantity]

    description = 'Rohde and Schwarz synthesized signal generator'

    def __init__(self, name, GPIB_address, **kwargs):
        Device.__init__(self, name, None, 'GPIB', **kwargs)

        self.instructions = {}

        self.GPIB_address = GPIB_address
        self.BLACS_connection = self.GPIB_address

    def generate_code(self, hdf5_file):
        group = self.init_device_group(hdf5_file)


import os
from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED

from blacs.device_base_class import DeviceTab

from qtutils import UiLoader
from qtutils.qt.QtCore import *
from qtutils.qt.QtGui import QDoubleValidator


@BLACS_tab
class EIP_545ATab(DeviceTab):

    def initialise_GUI(self):
        self.GPIB_address = str(self.settings['connection_table'].find_by_name(self.device_name).BLACS_connection)

    def initialise_workers(self):
        worker_initialisation_kwargs = {'GPIB_address': self.GPIB_address}
        self.create_worker("main_worker", EIP_545AWorker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"

    @define_state(MODE_MANUAL, True)
    def transition_to_buffered(self, h5_file, notify_queue):
        DeviceTab.transition_to_buffered(self, h5_file, notify_queue)

    @define_state(MODE_BUFFERED, False)
    def transition_to_manual(self, notify_queue, program=False):
        DeviceTab.transition_to_manual(self, notify_queue, program)


from labscript_devices.GPIBDevice import GPIBWorker


class EIP_545AWorker(GPIBWorker):

    def __init__(self, *args, **kwargs):
        self.h5_file = None
        self.device_name = ""
        GPIBWorker.__init__(self, *args, **kwargs)

    def program_manual(self, *args, **kwargs):
        pass

    def transition_to_manual(self, *args, **kwargs):
        freq = float(self.GPIB_connection.query('B3R0FAFRH'))

        with h5py.File(self.h5_file, 'a') as hdf5_file:
            data_group = hdf5_file['data']
            data_group.create_dataset(self.device_name, data=np.array(freq))

        return True

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        self.h5_file = h5file
        self.device_name = device_name
        self.GPIB_connection.read_termination = '\r'
        return {}


@runviewer_parser
class RunviewerClass(object):
    pass
