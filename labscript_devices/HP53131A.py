from labscript_devices import runviewer_parser, BLACS_tab
from labscript import config, Device, IntermediateDevice, StaticAnalogQuantity, LabscriptError, set_passed_properties
import numpy as np
import labscript_utils.h5_lock
import h5py
import threading


class HP53131A(IntermediateDevice):
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
class HP53131ATab(DeviceTab):

    def initialise_GUI(self):
        self.GPIB_address = str(self.settings['connection_table'].find_by_name(self.device_name).BLACS_connection)

    def initialise_workers(self):
        worker_initialisation_kwargs = {
            'GPIB_address': self.GPIB_address,
            'jump_address': str(self.settings['connection_table'].jump_device_address),
        }
        self.create_worker("main_worker", HP53131AWorker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"

    @define_state(MODE_MANUAL, True)
    def transition_to_buffered(self, h5_file, notify_queue):
        DeviceTab.transition_to_buffered(self, h5_file, notify_queue)

    @define_state(MODE_BUFFERED, False)
    def transition_to_manual(self, notify_queue, program=False):
        DeviceTab.transition_to_manual(self, notify_queue, program)


from labscript_devices.GPIBDevice import GPIBWorker


class HP53131AWorker(GPIBWorker):

    def __init__(self, *args, **kwargs):
        self.h5_file = None
        self.device_name = ""
        GPIBWorker.__init__(self, *args, **kwargs)

    def program_manual(self, *args, **kwargs):
        pass

    def transition_to_manual(self, *args, **kwargs):
        if self.run_thread is not None:
            self.run_thread.join()
        freq = self.GPIB_connection.query(":READ:FREQ?")

        with h5py.File(self.h5_file, 'a') as hdf5_file:
            data_group = hdf5_file['data']
            data_group.create_dataset(self.device_name, data=np.array(float(freq)))

        return True


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
        self.GPIB_connection.write(":FUNC 'FREQ 1'")
        self.h5_file = h5file
        self.device_name = device_name

        # start run thread
        self.run_thread = threading.Thread(target=self.run_experiment, args=(device_name,))
        self.run_thread.start()

        return {}


@runviewer_parser
class RunviewerClass(object):
    pass
