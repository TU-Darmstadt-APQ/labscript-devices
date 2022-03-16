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


class AndorGridJumper(Device):
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
class AndorGridJumperTab(DeviceTab):
    def initialise_GUI(self):
        layout = self.get_tab_layout()
        ui_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'andor_grid_jumper.ui')
        self.ui = UiLoader().load(ui_filepath)
        layout.addWidget(self.ui)

        port = int(self.settings['connection_table'].find_by_name(self.settings["device_name"]).BLACS_connection)
        self.ui.port_label.setText(str(port))

        self.ui.check_connectivity_pushButton.setIcon(QIcon(':/qtutils/fugue/arrow-circle'))

        self.ui.host_lineEdit.returnPressed.connect(self.update_settings_and_check_connectivity)
        self.ui.check_connectivity_pushButton.clicked.connect(self.update_settings_and_check_connectivity)

    def get_save_data(self):
        return {'host': str(self.ui.host_lineEdit.text())}

    def restore_save_data(self, save_data):
        if save_data:
            host = save_data['host']
            self.ui.host_lineEdit.setText(host)
        else:
            self.logger.warning('No previous front panel state to restore')

        # call update_settings if primary_worker is set
        # this will be true if you load a front panel from the file menu after the tab has started
        if self.primary_worker:
            self.update_settings_and_check_connectivity()

    def initialise_workers(self):
        worker_initialisation_kwargs = {
            'port': self.ui.port_label.text(),
            'jump_address': str(self.settings['connection_table'].jump_device_address),
            'devices': json.dumps([str(d) for d in self.settings["connection_table"].get_attached_devices().keys()]),
        }
        self.create_worker("main_worker", AndorGridJumperWorker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"
        self.update_settings_and_check_connectivity()

    @define_state(MODE_MANUAL, queue_state_indefinitely=True, delete_stale_states=True)
    def update_settings_and_check_connectivity(self, *args):
        icon = QIcon(':/qtutils/fugue/hourglass')
        pixmap = icon.pixmap(QSize(16, 16))
        status_text = 'Checking...'
        self.ui.status_icon.setPixmap(pixmap)
        self.ui.server_status.setText(status_text)
        kwargs = self.get_save_data()
        responding = yield(self.queue_work(self.primary_worker, 'update_settings_and_check_connectivity', **kwargs))
        self.update_responding_indicator(responding)

    def update_responding_indicator(self, responding):
        if responding:
            icon = QIcon(':/qtutils/fugue/tick')
            pixmap = icon.pixmap(QSize(16, 16))
            status_text = 'Server is responding'
        else:
            icon = QIcon(':/qtutils/fugue/exclamation')
            pixmap = icon.pixmap(QSize(16, 16))
            status_text = 'Server not responding'
        self.ui.status_icon.setPixmap(pixmap)
        self.ui.server_status.setText(status_text)

    @define_state(MODE_BUFFERED, True)
    def start_run(self, notify_queue):
        """This starts the run."""
        yield (self.queue_work(self.primary_worker, "start_run"))


class AndorGridJumperWorker(Worker):
    def init(self):

        self.runner = RunMasterClass()
        self.runner.start()

        self.runner.set_compute_next_section_callback(self.next_section)

        self.jump_counter = 0
        self.sections = []
        self.current_section = 0
        self.total_past_t = 0

    def program_manual(self, values):
        return {}

    def shutdown(self):
        self.runner.shutdown()

    def update_settings_and_check_connectivity(self, host):
        self.host = host
        if not self.host:
            return False

        self.get_grid()
        return True

    def get_grid(self):
        print("GET GRID")
        try:
            grid = zprocess.zmq_get(self.port, self.host, 'get_grid', 1.5)
            grid[:, 0:4] = False
            # grid = []
            print("GOT GRID")
            return grid
        except:
            # need to add: Make a Fake Exposure
            raise Exception('No grid received')

    def start_run(self):
        self.runner.send_start()

    def next_section(self):
        max_jumps = 20

        # Evaluate which section is next
        next_section = self.current_section + 1

        # Must be called every time currently... :/
        grid = self.get_grid()
        print(grid)

        if self.sections[self.current_section]['jump']:
            # TODO: evaluate where to jump. Currently just jump 5 times.

            print("Jump decision")

            # check all positions
            res_grid = [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
            ]
            grid_full = np.all(grid & res_grid == res_grid)
            print(f"Grid full: {grid_full}")

            should_jump = not grid_full
            if self.sections[self.current_section]['inverted']:
                print("INVERTED",)
                should_jump = not should_jump

            if should_jump and self.jump_counter < max_jumps:
                next_section = self.sections[self.current_section]['to_section']
                self.jump_counter += 1

        print("next section: ", next_section)
        print("jump_counter: ", self.jump_counter)

        print(len(self.sections))
        if next_section >= len(self.sections):
            print("exit!")
            return -1
        else:
            self.current_section = next_section
            return next_section

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):

        self.current_section = 0
        self.total_past_t = 0
        self.jump_counter = 0

        with h5py.File(h5file, 'r') as hdf5_file:
            jumps = hdf5_file['jumps'][:]
            master_clock = hdf5_file['connection table'].attrs['master_pseudoclock']
            end_time = hdf5_file['devices'][master_clock].attrs['stop_time']

        timestamps = []
        for j in range(len(jumps)):
            timestamps.append(jumps[j]["time"])
            timestamps.append(jumps[j]["to_time"])

        timestamps.append(0)
        timestamps.append(end_time)

        timestamps = sorted(set(timestamps))

        self.sections = []
        for i in range(len(timestamps) - 1):
            start = timestamps[i]
            end = timestamps[i + 1]
            jump = False
            to_time = 0
            for i in range(len(jumps)):
                if end - 0.00001 < jumps[i]['time'] < end + 0.00001:
                    jump = True
                    to_time = jumps[i]['to_time']
                    break
            section = {
                "start": start,
                "end": end,
                "jump": jump,
                "inverted": False,
                "to_time": to_time,
            }

            self.sections.append(section)

        self.sections[0]["inverted"] = True  # TODO: move to experiment file

        for i in range(len(self.sections)):
            to_time = self.sections[i]['to_time']
            to_section = i
            for j in range(len(self.sections)):
                if to_time - 0.00001 < self.sections[j]['start'] < to_time + 0.00001:
                    to_section = j
                    break

            self.sections[i]['to_section'] = to_section

        print(self.sections)

        self.runner.send_buffered()

        return {}

    def transition_to_manual(self, abort=False):
        return True

    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)

    def abort_buffered(self):
        self.runner.abort()
        return self.transition_to_manual(True)
