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

        self.last_grid = [[]]

        self.section_history = []
        self.jump_history = []

        self.next_section_time = []

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

    def get_grid(self, reuse_grid=False, times=1):
        if reuse_grid:
            return self.last_grid

        try:
            for i in range(times):  # HACK TO REMOVE TOO MANY GRIDS...
                grid = zprocess.zmq_get(self.port, self.host, 'get_grid', 1.5)
            #grid[:, 0:4] = False

            self.last_grid = grid

            return grid
        except:
            # need to add: Make a Fake Exposure
            raise Exception('No grid received')

    def start_run(self):
        self.runner.send_start()

    def next_section(self):
        # Evaluate which section is next
        start_t = time.perf_counter()
        next_section = self.current_section + 1

        jump_decision = {
            'executed_jump': False,
            'jump_label': self.sections[self.current_section]['jump_label'],
            'prev_jump_count': self.sections[self.current_section]['jump_counter'],
            'is_jump': False,
            'jump_cond': False,
            'jump_count_limit': self.sections[self.current_section]['max_jumps']
        }

        if self.sections[self.current_section]['is_jump']:

            should_jump = False
            if self.sections[self.current_section]['dummy']:
                should_jump = True
            else:
                grid = self.get_grid(self.sections[self.current_section]['reuse_grid'], self.sections[self.current_section]['get_grid_times'])
                res_grid = self.sections[self.current_section]['grid']

                if self.sections[self.current_section]['negative_grid']:
                    grid_full = np.any(grid)  # TODO
                else:
                    grid_full = np.all(grid & res_grid == res_grid)

                should_jump = grid_full
                if self.sections[self.current_section]['inverted']:
                    should_jump = not should_jump

            jump_decision['jump_cond'] = should_jump
            if should_jump and self.sections[self.current_section]['jump_counter'] < self.sections[self.current_section]['max_jumps']:
                jump_decision['executed_jump'] = True
                next_section = self.sections[self.current_section]['to_section']
                self.sections[self.current_section]['jump_counter'] += 1

        self.section_history.append(next_section)
        self.jump_history.append(jump_decision)

        t_next_section = time.perf_counter() - start_t
        self.next_section_time.append(t_next_section)
        if next_section >= len(self.sections):
            return -1
        else:
            self.current_section = next_section
            return next_section

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):

        self.h5 = h5file

        self.current_section = 0
        self.section_history = [0]
        self.jump_history = []

        self.next_section_time = []

        self.transition_times = []
        self.run_times = []
        self.last_grid = [[]]

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

            jump = None
            for i in range(len(jumps)):
                if end - 0.00001 < jumps[i]['time'] < end + 0.00001:
                    jump = jumps[i]
                    break

            if jump is not None:
                jump_data = json.loads(jump['data'])
                inverted = False
                if 'inverted' in jump_data:
                    inverted = jump_data['inverted']
                dummy = False
                if 'dummy' in jump_data:
                    dummy = jump_data['dummy']
                reuse_grid = False
                if 'reuse_grid' in jump_data:
                    reuse_grid = jump_data['reuse_grid']
                get_grid_times = 1
                if 'get_grid_times' in jump_data:
                    get_grid_times = jump_data['get_grid_times']
                negative_grid = False
                if 'negative_grid' in jump_data:
                    negative_grid = jump_data['negative_grid']
                grid = jump_data['grid']

                section = {
                    "start": start,
                    "end": end,
                    "is_jump": True,
                    "jump_label": jump['label'],
                    "inverted": inverted,
                    "reuse_grid": reuse_grid,
                    "max_jumps": jump['max_jumps'],
                    "get_grid_times": get_grid_times,
                    "jump_counter": 0,
                    "negative_grid": negative_grid,
                    "grid": np.array(grid),
                    "to_time": jump['to_time'],
                    "dummy": dummy
                }
            else:
                section = {
                    "start": start,
                    "end": end,
                    "is_jump": False,
                    "jump_label": 'None',
                    "inverted": False,
                    "reuse_grid": False,
                    "negative_grid": False,
                    "get_grid_times": 0,
                    "max_jumps": 0,
                    "jump_counter": 0,
                    "to_time": 0,
                    "dummy": False
                }

            self.sections.append(section)

        for i in range(len(self.sections)):
            to_time = self.sections[i]['to_time']
            to_section = i
            for j in range(len(self.sections)):
                if to_time - 0.00001 < self.sections[j]['start'] < to_time + 0.00001:
                    to_section = j
                    break

            self.sections[i]['to_section'] = to_section

        self.runner.send_buffered()

        return {}

    def transition_to_manual(self, abort=False):

        if abort:
            return True

        sections_dtypes = [
            ('id', int),
            ('transition_time', float),
            ('run_time', float)
        ]
        jumps_dtypes = [
            ('executed_jump', bool),
            ('jump_label', 'a256'),
            ('prev_jump_count', int),
            ('is_jump', bool),
            ('jump_cond', bool),
            ('jump_count_limit', int),
        ]

        # We ignore the last section as this is section "-1"
        sections_data = np.empty(len(self.section_history) - 1, dtype=sections_dtypes)
        for i in range(len(self.section_history) - 1):
            sections_data[i]['id'] = self.section_history[i]
            sections_data[i]['transition_time'] = self.transition_times[i]
            sections_data[i]['run_time'] = self.run_times[i]

        jumps_data = np.empty(len(self.jump_history), dtype=jumps_dtypes)
        for i in range(len(self.jump_history)):
            jumps_data[i] = self.jump_history[i]['executed_jump'], self.jump_history[i]['jump_label'], self.jump_history[i]['prev_jump_count'], self.jump_history[i]['is_jump'], self.jump_history[i]['jump_cond'], self.jump_history[i]['jump_count_limit']

        with h5py.File(self.h5, 'a') as hdf5_file:
            group = hdf5_file['data']
            group.create_dataset('section_history', data=sections_data)
            group.create_dataset('jump_history', data=jumps_data)
            group.create_dataset('time_next_section', data=self.next_section_time)

        return True

    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)

    def abort_buffered(self):
        self.runner.abort()
        return self.transition_to_manual(True)
