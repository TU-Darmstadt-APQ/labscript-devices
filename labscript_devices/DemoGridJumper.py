from tracemalloc import start
import zmq
import json
import time
import threading
import os

from labscript_devices import BLACS_tab
from labscript import Device, LabscriptError, set_passed_properties
import numpy as np

import zprocess

import labscript_utils.h5_lock
import h5py

from labscript_utils.in_exp_com import RunMasterClass


class DemoGridJumper(Device):
    @set_passed_properties(
        property_names={
            "connection_table_properties": ["BIAS_port"]}
    )
    def __init__(self, name, BIAS_port=1027, **kwargs):
        Device.__init__(self, name=name, parent_device=None, connection=None, **kwargs)
        self.BLACS_connection = BIAS_port

    def generate_code(self, hdf5_file):
        pass


from qtutils.qt.QtCore import *
from qtutils.qt.QtGui import *

from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED

from blacs.device_base_class import DeviceTab

from qtutils import UiLoader
import qtutils.icons

RUNNING = "run"
FINISHED = "fin"
LOADING = "lod"
READY = "rdy"
EXITED = "ext"


@BLACS_tab
class DemoGridJumperTab(DeviceTab):
    def initialise_GUI(self):
        layout = self.get_tab_layout()
        ui_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'andor_grid_jumper.ui')
        self.ui = UiLoader().load(ui_filepath)
        layout.addWidget(self.ui)

    def get_save_data(self):
        return {}

    def restore_save_data(self, save_data):
        pass

    def initialise_workers(self):
        worker_initialisation_kwargs = {
            'port': self.ui.port_label.text(),
            'jump_address': str(self.settings['connection_table'].jump_device_address),
            'devices': json.dumps([str(d) for d in self.settings["connection_table"].get_attached_devices().keys()]),
        }
        self.create_worker("main_worker", DemoGridJumperWorker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"

    @define_state(MODE_BUFFERED, True)
    def start_run(self, notify_queue):
        """This starts the run."""
        yield (self.queue_work(self.primary_worker, "start_run"))


class DemoGridJumperWorker(Worker):
    def init(self):

        self.h5 = None

        self.transition_times = []
        self.run_times = []

        self.runner = RunMasterClass()
        self.runner.start()

        self.runner.set_compute_next_section_callback(self.next_section)
        self.runner.set_transition_time_callback(lambda x: self.transition_times.append(x))
        self.runner.set_run_time_callback(lambda x: self.run_times.append(x))

        self.sections = []
        self.current_section = 0

        self.section_history = []
        self.jump_history = []

        self.next_section_time = []

    def program_manual(self, values):
        return {}

    def shutdown(self):
        self.runner.shutdown()

    def start_run(self):
        self.runner.send_start()

    def next_section(self):
        return -1

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):

        self.runner.send_buffered()

        return {}

    def transition_to_manual(self, abort=False):
        return True

    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)

    def abort_buffered(self):
        self.runner.abort()
        return self.transition_to_manual(True)
