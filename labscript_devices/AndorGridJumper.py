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


class AndorGridJumperWorker(Worker):
    def init(self):

        self.context = zmq.Context()

        self.devices = json.loads(self.devices)
        self.devices = [str.encode(d) for d in self.devices]

        self.from_master_socket = self.context.socket(zmq.PUB)
        self.from_master_socket.bind(f"tcp://*:44555")
        self.to_master_socket = self.context.socket(zmq.SUB)
        self.to_master_socket.bind(f"tcp://*:44556")

        self.to_master_socket.subscribe("")

        self.sections = []
        self.current_section = 0
        self.total_past_t = 0

        self.device_states = {}
        for dev in self.devices:
            self.device_states[dev] = EXITED

        self.jump_thread = None

    def program_manual(self, values):
        return {}

    def shutdown(self):

        self.from_master_socket.close()
        self.to_master_socket.close()
        self.context.term()

    def update_settings_and_check_connectivity(self, host):
        self.host = host
        if not self.host:
            return False

        self.get_grid()
        return True

    def get_grid(self):
        try:
            #rid = zprocess.zmq_get(self.port, self.host, 'get_grid', 0.5)
            # grid[:,0:4]=False
            grid = []
            return grid
        except:
            # need to add: Make a Fake Exposure
            raise Exception('No grid received')

    def run_experiment(self, device_name):

        print("Run experiment")
        max_jumps = 5
        jump_counter = 0

        time.sleep(0.5)  # TODO: REMOVE!!!

        self.from_master_socket.send(b"init")

        while True:

            start_time = time.perf_counter()

            # Wait for all devices to finish
            self.device_states[str.encode(device_name)] = FINISHED  # This device is always finished
            print(f"Finished {device_name}")
            while True:
                msg = self.to_master_socket.recv()
                device = msg.split()[-1]
                self.device_states[device] = FINISHED

                is_done = True
                for dev in self.device_states:
                    if self.device_states[dev] != FINISHED:
                        print(f"Waiting for {dev}")
                        is_done = False
                if is_done:
                    break

                print(f"")

            # Evaluate which section is next
            next_section = self.current_section + 1
            if self.sections[self.current_section]['jump']:
                # TODO: evaluate where to jump. Currently just jump 5 times.

                print("Jump decision")

                grid = self.get_grid()
                print(grid)

                if jump_counter < max_jumps:
                    jump_counter += 1
                    for s in range(len(self.sections)):
                        if self.sections[s]['start'] == self.sections[self.current_section]['to_time']:
                            next_section = s
                            break

            if next_section >= len(self.sections):
                self.from_master_socket.send(b"exit")
                break  # Shot finished

            self.current_section = next_section

            print(f"load {next_section}")
            # Prepare all devices
            self.from_master_socket.send(str.encode(f"load {next_section}"))

            # Wait for all devices to be ready
            self.device_states[str.encode(device_name)] = READY  # This device is always finished
            while True:
                msg = self.to_master_socket.recv()
                device = msg.split()[-1]
                self.device_states[device] = READY

                is_done = True
                for dev in self.device_states:
                    if self.device_states[dev] != READY:
                        is_done = False
                if is_done:
                    break

            # Send start signal
            self.from_master_socket.send(b"start")

            finish_time = time.perf_counter()
            t = finish_time - start_time
            print(f"Loop took {t*1e6:.2f}us")

            for dev in self.device_states:
                self.device_states[dev] = RUNNING

        print("Finished experiment...")

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):

        self.current_section = 0
        self.total_past_t = 0

        for dev in self.devices:
            self.device_states[dev] = RUNNING

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
                if jumps[i]['time'] == end:
                    jump = True
                    to_time = jumps[i]['to_time']
                    break
            section = {
                "start": start,
                "end": end,
                "jump": jump,
                "to_time": to_time
            }

            self.sections.append(section)

        self.jump_thread = threading.Thread(target=self.run_experiment, args=(device_name,))
        self.jump_thread.start()

        return {}

    def transition_to_manual(self, abort=False):

        # TODO: stop run thread

        self.device_states = {}
        for dev in self.devices:
            self.device_states[dev] = EXITED

        return True

    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)

    def abort_buffered(self):
        return self.transition_to_manual(True)
