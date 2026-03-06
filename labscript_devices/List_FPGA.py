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
import h5py
import time


class List_FPGA(TriggerableDevice):
    NUM_CHN_MAX = 6
    NUM_IOS = 3
    description = 'List based FPGA '

    # DDS_OUTPUT_STATE = None  # is set via __init__ keeps track of cuurent output state

    # Defined trigger values:
    LIST_FPGA_TRIGGER_WAIT_ANY = 0
    LIST_FPGA_TRIGGER_WAIT_HARDWARE = 1
    LIST_FPGA_TRIGGER_WAIT_SOFTWARE = 2
    LIST_FPGA_TRIGGER_NO_WAIT = 3

    # Dummy DDS:
    DDS_NONE = 1000  # No DDS frequency update is done for any instruction with dds == DDS_NONE.
    DDS_NONE_FREQ = 377.5e6  # No DDS frequency update is done for any instruction with dds == DDS_NONE. but a frequency has to appendted to the instructions

    def __init__(self,
                 name,
                 parent_device,
                 connection,
                 IP_address,
                 freq_base_min_s,
                 freq_base_max_s,
                 freq_m_s=None,
                 freq_b_s=None,
                 freq_base_default=None,
                 **kwargs):
        if freq_m_s is None:
            freq_m_s = [1.0] * self.NUM_CHN_MAX
        if freq_b_s is None:
            freq_b_s = [0.0] * self.NUM_CHN_MAX
        if freq_base_default is None:
            freq_base_default = freq_base_min_s
        self.trigger_edge_type = 'rising'
        self.BLACS_connection = IP_address + "::" + str(freq_base_min_s) + "::" + str(freq_base_max_s) + "::" + str(freq_m_s) + "::" + str(freq_b_s) + "::" + str(freq_base_default)
        self.instructions = []

        self.DDS_OUTPUT_STATE = int("000", 2)  # default everything is off, keeps track of the current dds state
        self.DDS_OUTPUT_STATE_LIST = [0, 0, 0]

        self.name = name
        TriggerableDevice.__init__(self, name, parent_device, connection, **kwargs)

    def trigger(self, t):
        return self.trigger_device.trigger(t, 2e-6)

    def create_fifo_entry(self, freq, dds, delay, trigger_mode, digitalOut):
        if (delay < 0):
            print("invalid delay < 0")
        elif (trigger_mode < 0 and trigger_mode > 3):
            print("invalid trigger_mode")
        elif (dds != self.DDS_NONE) and (dds < 0 or dds > 5):
            print("invalid DDS number")
        elif freq < 0:
            print("invalid frequency < 0")
        elif digitalOut < 0 or digitalOut > 7:
            print("invalid digital port state")
        else:
            return [freq, dds, delay, trigger_mode, digitalOut]

    def add_instructions(self, frequency, DDS_Nr, Delay, Trigger_type, digital_out):
        """
        frequency, DDS_Nr, Delay, Trigger_type

        Trigger_type = 1 (Hardware Trigger), 2 (Software Trigger), 3 (Internal Trigger)
        digital_out: 3-bit number where each bit defines whether the port is set to high (1) or "low" (0) after the specified delay together with DDS tx_enable.
        """
        self.DDS_OUTPUT_STATE = digital_out
        self.instructions.append([np.float32(frequency),
                                  np.int32(DDS_Nr),
                                  np.int32(Delay),
                                  np.int32(Trigger_type),
                                  np.int32(digital_out)])

    def add_only_frequency(self, frequency, DDS_Nr, Delay, Trigger_type):
        """
        Same as self.add_instructions but does not change the io-state
        """
        self.instructions.append([np.float32(frequency),
                                  np.int32(DDS_Nr),
                                  np.int32(Delay),
                                  np.int32(Trigger_type),
                                  np.int32(self._int_for_switch())])

    def add_switch(self, delay, trigger_type, digital_out_id, digital_out_val):
        """
        trigger_type:
            = 1 (Hardware Trigger), 2 (Software Trigger), 3 (Internal Trigger)
        digital_out_id:
            number of output to use
        digital_out_val: bool
            value of the digital output "high" (1) or "low" (0)
        """
        if (0 <= digital_out_id and digital_out_id < self.NUM_IOS) == False:
            raise ValueError(f"Invalid FPGA-IO-Pin used: you provided digital_out_id={digital_out_id}\
                             but only supported io-pin_numbers are: 0 <= digital_out_id < self.NUM_IOS={self.NUM_IOS}")
        rev_mask_str = ""
        for i in range(0, self.NUM_IOS):
            if i == digital_out_id:
                rev_mask_str += str(int(digital_out_val))
            else:
                rev_mask_str += "1"

        mask_str = rev_mask_str[::-1]
        mask = int(mask_str, 2)
        self.DDS_OUTPUT_STATE = self.DDS_OUTPUT_STATE & mask  # | (bool(digital_out_val) << digital_out_id)

        for i in range(0, len(self.DDS_OUTPUT_STATE_LIST)):
            if i == digital_out_id:
                self.DDS_OUTPUT_STATE_LIST[i] = int(bool(digital_out_val))
                break

        self.instructions.append([np.float32(self.DDS_NONE_FREQ),
                                  np.int32(self.DDS_NONE),
                                  np.int32(delay),
                                  np.int32(trigger_type),
                                  np.int32(self._int_for_switch())])
        # print("fpga", digital_out_id, digital_out_val, self.DDS_OUTPUT_STATE, mask, self.DDS_OUTPUT_STATE_LIST, v_int)
        pass

    def _int_for_switch(self):
        v_str = ""
        for j in range(0, len(self.DDS_OUTPUT_STATE_LIST)):
            v_str += str(self.DDS_OUTPUT_STATE_LIST[j])

        v_int = int(v_str[::-1], 2)
        # print(self.DDS_OUTPUT_STATE_LIST)
        return v_int
        pass

    def software_wait_trigger(self, delay=0):
        self.instructions.append([np.float32(self.DDS_NONE_FREQ),
                                  np.int32(self.DDS_NONE),
                                  np.int32(delay),
                                  np.int32(self.LIST_FPGA_TRIGGER_WAIT_SOFTWARE),
                                  np.int32(self.DDS_OUTPUT_STATE)])

    def generate_code(self, hdf5_file):
        group = self.init_device_group(hdf5_file)

        group.create_dataset(f'Instructions', data=self.instructions)


import os

from qtutils.qt.QtCore import *
from qtutils.qt.QtGui import *

from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED

from blacs.device_base_class import DeviceTab

from qtutils import UiLoader
import qtutils.icons


@BLACS_tab
class List_FPGATab(DeviceTab):
    NUM_CHN_MAX = 6

    def initialise_GUI(self):
        layout = self.get_tab_layout()
        ui_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'FPGA.ui')
        self.ui = UiLoader().load(ui_filepath)
        self.ui.formGroupBox.setTitle('List FPGA System')
        self.ui.use_zmq_checkBox.hide()
        layout.addWidget(self.ui)

        self.ui.check_connectivity_pushButton.setIcon(QIcon(':/qtutils/fugue/arrow-circle'))

        self.ui.use_zmq_checkBox.toggled.connect(self.update_settings_and_check_connectivity)
        self.ui.check_connectivity_pushButton.clicked.connect(self.update_settings_and_check_connectivity)

        connection_table_entry = self.settings['connection_table'].find_by_name(self.settings['device_name'])

        self.IP_address = connection_table_entry.BLACS_connection
        print("self.IP_address = {}".format(self.IP_address))
        self.host = self.IP_address.split("::")[0]
        self.port = self.IP_address.split("::")[1]
        self.ui.host_label.setText(self.host)
        self.ui.port_label.setText(self.port)

        """# old Capabilities
                                self.f_base_units = 'MHz'  # front panel values are in GHz!
                                self.f_base_min = 3035.73
                                self.f_base_max = 3235.73
                                self.f_base_step = 20 # step size for +/- buttons
                                self.f_base_decimals = 5  # display 6 decimals accuracy"""
        # new Capabilities
        self.f_base_units = 'MHz'  # front panel values are in GHz!
        self.f_base_decimals = 5  # display 6 decimals accuracy
        # print("self.IP_address[2][1:-1].split(',') = {}".format(self.IP_address.split("::")[2][1:-1].split(",")))
        self.f_base_min_s = list(map(lambda x: float(x), self.IP_address.split("::")[2][1:-1].split(",")))  # [3034.73, 90.00, 90.00, 30.00, 30.00, 30.00]
        print("self.f_base_min_s = {}".format(self.f_base_min_s))
        self.f_base_max_s = list(map(lambda x: float(x), self.IP_address.split("::")[3][1:-1].split(",")))  # [3235.73, 130.00, 130.00, 70.00, 70.00, 70.00]
        self.f_m_s = list(map(lambda x: float(x), self.IP_address.split("::")[4][1:-1].split(",")))
        self.f_b_s = list(map(lambda x: float(x), self.IP_address.split("::")[5][1:-1].split(",")))
        self.f_base_step_s = [1, 1, 1, 1, 1, 1]  # step size for +/- buttons

        # old properties
        analog_properties = {}
        """
        analog_properties['frequency'] = {'base_unit': self.f_base_units,
                                                                  'min': self.f_base_min,
                                                                  'max': self.f_base_max,
                                                                  'step': self.f_base_step,
                                                                  'decimals': self.f_base_decimals
                                                                  }"""
        # new properties
        for i in range(0, self.NUM_CHN_MAX):
            analog_properties[f"frequency {i}"] = {'base_unit': self.f_base_units,
                                                   'min': self.f_base_min_s[i],
                                                   'max': self.f_base_max_s[i],
                                                   'step': self.f_base_step_s[i],
                                                   'decimals': self.f_base_decimals
                                                   }

        self.create_analog_outputs(analog_properties)

        _, ao_widgets, _ = self.auto_create_widgets()

        self.auto_place_widgets(ao_widgets)

    def initialise_workers(self):
        worker_initialisation_kwargs = {'IP_address': self.IP_address}
        self.create_worker("main_worker", ListFPGAWorker_ZMQ, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"
        self.update_settings_and_check_connectivity()

    @define_state(MODE_MANUAL, queue_state_indefinitely=True, delete_stale_states=True)
    def update_settings_and_check_connectivity(self, *args):
        icon = QIcon(':/qtutils/fugue/hourglass')
        pixmap = icon.pixmap(QSize(16, 16))
        status_text = 'Checking...'
        self.ui.status_icon.setPixmap(pixmap)
        self.ui.server_status.setText(status_text)
        kwargs = {'host': self.host, 'port': self.port}
        responding = yield(self.queue_work(self.primary_worker, 'update_settings_and_check_connectivity', **kwargs))
        self.update_responding_indicator(responding)

    def update_responding_indicator(self, responding):
        if responding:
            icon = QIcon(':/qtutils/fugue/tick')
            pixmap = icon.pixmap(QSize(16, 16))
            status_text = 'FPGA is responding'
        else:
            icon = QIcon(':/qtutils/fugue/exclamation')
            pixmap = icon.pixmap(QSize(16, 16))
            status_text = 'FPGA not responding'
        self.ui.status_icon.setPixmap(pixmap)
        self.ui.server_status.setText(status_text)


class ListFPGAWorker_ZMQ(Worker):
    NUM_CHN_MAX = 6

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
        global struct
        import struct

        self.host = 'localhost'
        self.port = '5555'

    def update_settings_and_check_connectivity(self, host, port):
        self.host = host
        self.port = port
        if not self.host:
            return False
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect('tcp://{host}:{port}'.format(host=str(self.host), port=str(self.port)))
        try:
            message = self.pack_data([3, 0], 'Header')
            print(message)
            self.send_data(message)
            return True
        except:
            raise Exception('no response')

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        self.reset_fifo()

        FIFO_ENTRY = 0
        value_list = []
        with h5py.File(h5file, 'r') as hdf5_file:
            group = hdf5_file['globals']
            # frequencies = group.attrs['raman_mf0_state_prep_frequencies']
            # pulse_durations = group.attrs['raman_mf0_pulse_durations']
            group = hdf5_file['devices'][device_name]
            for instr in group['Instructions']:
                value_list.append([np.float32(instr[0]), np.int(instr[1]), np.int(instr[2]), np.int(instr[3]), np.int(instr[4])])

        header = self.pack_data([FIFO_ENTRY, len(value_list)], 'Header')
        FPGA_data = self.pack_data(value_list, 'List')
        # print(f"value_list = {value_list}")
        self.send_data(header, flag=zmq.SNDMORE)
        self.send_data(FPGA_data)
        # self.send_trigger()

        return {}  # indicates final values of buffered run, we have none

    def pack_data(self, data, data_identifier):
        if data_identifier == 'List':
            list_2join = []
            for data_entry in data:
                try:
                    list_2join = struct.pack('<fIIII', *data_entry)
                except Exception as e:
                    raise Exception(f"Eror while handling pack instruction: data_entry = {data_entry}, Exception={e}")

            packed_data = b"".join([struct.pack('<fIIII', *data_entry) for data_entry in data])
        elif data_identifier == 'Header':
            packed_data = struct.pack('<II', *data)
        else:
            raise Exception('No correct data_identifier was given')
        return packed_data

    def transition_to_manual(self):
        return True

    def program_manual(self, front_panel_values):
        print(f"{time.time()}: ListFPGAWorker_ZMQ.program_manual() got called")
        # print(f"self.IP_address = {self.IP_address}")
        instructions = [[]] * self.NUM_CHN_MAX
        f_base_min_s = list(map(lambda x: float(x), self.IP_address.split("::")[2][1:-1].split(",")))  # hacky way of transfering data between the classes without editing labscript.py...
        f_base_max_s = list(map(lambda x: float(x), self.IP_address.split("::")[3][1:-1].split(",")))
        freq_m_s = list(map(lambda x: float(x), self.IP_address.split("::")[4][1:-1].split(",")))
        freq_b_s = list(map(lambda x: float(x), self.IP_address.split("::")[5][1:-1].split(",")))
        freq_base_default = list(map(lambda x: float(x), self.IP_address.split("::")[6][1:-1].split(",")))
        for i in range(0, self.NUM_CHN_MAX):

            if front_panel_values[f"frequency {i}"] * 1e6 * freq_m_s[i] + freq_b_s[i] >= f_base_min_s[i] * 1e6 and front_panel_values[f"frequency {i}"] * 1e6 * freq_m_s[i] + freq_b_s[i] <= f_base_max_s[i] * 1e6:
                this_frequency = front_panel_values[f"frequency {i}"] * 1e6 * freq_m_s[i] + freq_b_s[i]  # *1e6 for MHz to Hz --> m*x+b
            else:
                # expect error (because front_panel_values[f"frequency {i}"] is not in valid range)
                # use instead minimal value
                this_frequency = freq_base_default[i] * 1e6 * freq_m_s[i] + freq_b_s[i]

            this_instruction = [this_frequency, i, 3, 2, 0]  # frequency, DDS_Nr, Delay, Trigger_type, digital_out
            # if i == 0:
            #   instructions.append(this_instruction)
            instructions[i] = this_instruction
        data = self.pack_data(instructions, 'List')
        header = self.pack_data([0, len(instructions)], 'Header')
        print('Sending frequency(s): {} Hz'.format([f"DDS{sg_data[1]}: {sg_data[0]}" for sg_data in instructions]))
        self.send_data(header, flag=zmq.SNDMORE)
        self.send_data(data)
        self.send_trigger()
        return {}

    def send_data(self, message, flag=0):
        # socket.send('{}'.format(message).encode('utf-8'), flags = flag)
        self.socket.send(message, flag)  # zmq.NOBLOCK |
        if flag is not zmq.SNDMORE:
            response = self.socket.recv()
            resp = int.from_bytes(response, "little")
            if resp == 0:
                return True
            if resp == 1:
                raise Exception('FREQ_OUT_OF_RANGE; Obey strict larger and smaller for limits.')
            if resp == 2:
                raise Exception('DDS_OUT_OF_RANGE')
            if resp == 3:
                raise Exception('DELAY_OUT_OF_RANGE')
            if resp == 4:
                raise Exception('TRIGGER_MODE_OUT_OF_RANGE')
            if resp == 5:
                raise Exception('FIFO_ENTRY_SIZE_MISMATCH')
            if resp == 6:
                raise Exception('RAMP_SIZE_MISMATCH')
            if resp == 7:
                raise Exception('UNKNOWN_MESSAGE_TYPE')
            if resp == 8:
                raise Exception('INVALID_RAMP_TYPE')
            if resp == 9:
                raise Exception('INVALID_RAMP_SPEED')
            if resp == 10:
                raise Exception('UNEXPECTED_MULTI_PART')

    def abort_transition_to_buffered(self):
        return

    def abort_buffered(self):
        return True

    def shutdown(self):
        self.socket.close()
        return

    def send_trigger(self):
        SOFTWARE_TRIGGER = 2
        data_list = [SOFTWARE_TRIGGER, 0]
        data = struct.pack('<{}I'.format(len(data_list)), *data_list)
        self.socket.send(data, zmq.NOBLOCK)
        response = self.socket.recv()
        responseCode = int.from_bytes(response, "little")
        if (responseCode == 0):
            print("Success!")
        else:
            print("Error. Code:", responseCode)

    def reset_fifo(self):
        message = self.pack_data([4, 0], 'Header')
        self.send_data(message)
