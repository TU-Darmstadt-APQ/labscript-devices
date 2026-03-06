#####################################################################
#                                                                   #
# /labscript_devices/Tweezer.py                                     #
#                                                                   #
# Copyright 2018, Jan Werkmann                                      #
#                                                                   #
# This file is part of labscript_devices, in the labscript suite    #
# (see http://labscriptsuite.org), and is licensed under the        #
# Simplified BSD License. See the license.txt file in the root of   #
# the project for the full license.                                 #
#                                                                   #
#####################################################################

from labscript_devices import BLACS_tab
from labscript import TriggerableDevice, set_passed_properties
import numpy as np


class Tweezer_Offset(TriggerableDevice):
    description = 'Tweezer '

    # To be set as instantiation arguments:
    trigger_edge_type = None

    @set_passed_properties(
        property_names={
            "connection_table_properties": ["Tweezer_port"],
            "device_properties": ["ramp_speed", "amplitude", "rise_time", "fall_time", "wait_time", "routing_mode", "tweezer_mode", "offset"]}
    )
    def __init__(self, name, parent_device, connection,
                 Tweezer_port=1027, ramp_speed=1000, amplitude=5, res_grids=[], move_list=[], rise_time=300e-6, fall_time=300e-6, wait_time=100e-6,
                 routing_mode='default', tweezer_mode='rearrange', trigger_edge_type='rising', offset=(0, 0), **kwargs):

        self.trigger_edge_type = trigger_edge_type
        self.BLACS_connection = Tweezer_port

        self.res_grids = [res_grids] if isinstance(res_grids, np.ndarray) else res_grids
        self.move_list = [[pos1[0], pos1[1], pos2[0], pos2[1], amp] for pos1, pos2, amp in move_list]

        TriggerableDevice.__init__(self, name, parent_device, connection, **kwargs)

    def trigger(self, t):
        return self.trigger_device.trigger(t, 0.001)

    def generate_code(self, hdf5_file):
        group = self.init_device_group(hdf5_file)

        for i, grid in enumerate(self.res_grids):
            group.create_dataset('Grid{}'.format(i), data=grid)

        group.create_dataset('Moves', data=self.move_list)

        # DEPRECATED backward campatibility for use of exposuretime keyword argument instead of exposure_time:
        self.set_property('n_grids', len(self.res_grids), location='device_properties', overwrite=True)


import os

from qtutils.qt.QtCore import *
from qtutils.qt.QtGui import *

from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED

from blacs.device_base_class import DeviceTab

from qtutils import UiLoader
import qtutils.icons


@BLACS_tab
class TweezerTab(DeviceTab):
    def initialise_GUI(self):
        layout = self.get_tab_layout()
        ui_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'camera.ui')
        self.ui = UiLoader().load(ui_filepath)
        self.ui.formGroupBox.setTitle('Tweezer Control System')
        self.ui.use_zmq_checkBox.hide()
        layout.addWidget(self.ui)

        port = int(self.settings['connection_table'].find_by_name(self.settings["device_name"]).BLACS_connection)
        self.ui.port_label.setText(str(port))

        self.ui.check_connectivity_pushButton.setIcon(QIcon(':/qtutils/fugue/arrow-circle'))

        self.ui.host_lineEdit.returnPressed.connect(self.update_settings_and_check_connectivity)
        self.ui.use_zmq_checkBox.toggled.connect(self.update_settings_and_check_connectivity)
        self.ui.check_connectivity_pushButton.clicked.connect(self.update_settings_and_check_connectivity)

    def get_save_data(self):
        return {'host': str(self.ui.host_lineEdit.text())}

    def restore_save_data(self, save_data):
        print('restore save data running')
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
        worker_initialisation_kwargs = {'port': self.ui.port_label.text()}
        self.create_worker("main_worker", TweezerWorker, worker_initialisation_kwargs)
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


class TweezerWorker(Worker):
    def init(self):
        global zmq
        import zmq
        global zprocess
        import zprocess
        global shared_drive
        import labscript_utils.shared_drive as shared_drive
        global h5py
        import h5py
        global labscript_utils
        import labscript_utils

        self.host = ''

    def update_settings_and_check_connectivity(self, host):
        self.host = host
        if not self.host:
            return False

        response = zprocess.zmq_get(self.port, self.host, data=['hello'])
        if response == 'hello':
            return True
        else:
            raise Exception('invalid response from server: ' + str(response))

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        with h5py.File(h5file, 'r') as hdf5_file:
            device_properties = labscript_utils.properties.get(hdf5_file, device_name, 'device_properties')
            ramp_speed = device_properties['ramp_speed']
            amplitude = device_properties['amplitude']
            rise_time = device_properties['rise_time']
            fall_time = device_properties['fall_time']
            wait_time = device_properties['wait_time']
            tweezer_mode = device_properties['tweezer_mode']
            routing_mode = device_properties['routing_mode']
            offset = device_properties['offset']

            group = hdf5_file['devices/'][device_name]
            res_grids = [np.array(group.get('Grid{}'.format(i)), dtype=int) for i in range(device_properties['n_grids'])]
            move_list = [[(elenment[0], elenment[1]), (elenment[2], elenment[3]), int(elenment[4])] for elenment in np.array(group.get('Moves'))]  # deserialization
        settings_dict = {'ramp_speed': ramp_speed, 'amplitude': amplitude, 'res_grids': res_grids, 'rise_time': rise_time, 'fall_time': fall_time, 'wait_time': wait_time, 'move_list': move_list, 'tweezer_mode': tweezer_mode, 'routing_mode': routing_mode, 'offset': offset}
        request = ['to_buffered', settings_dict]
        response = zprocess.zmq_get(self.port, self.host, data=request)
        if response != 'ok':
            raise Exception('invalid response from server: ' + str(response))
        return {}  # indicates final values of buffered run, we have none

    def transition_to_manual(self):
        response = zprocess.zmq_get(self.port, self.host, ['to_manual'])
        if response != 'ok':
            raise Exception('invalid response from server: ' + str(response))
        return True

    def abort_buffered(self):
        return self.abort()

    def abort_transition_to_buffered(self):
        return self.abort()

    def abort(self):
        response = zprocess.zmq_get(self.port, self.host, ['abort'])
        if response != 'done':
            raise Exception('invalid response from server: ' + str(response))
        return True  # indicates success

    def program_manual(self, values):
        return {}

    def shutdown(self):
        return
