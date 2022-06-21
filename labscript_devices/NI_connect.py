try:
    from labscript_utils import check_version
except ImportError:
    raise ImportError('Require labscript_utils > 2.1.0')

check_version('labscript', '2.0.1', '3')

from labscript_devices import runviewer_parser, BLACS_tab
from labscript import config, IntermediateDevice, AnalogOut, LabscriptError, set_passed_properties
import numpy as np
import hashlib
import labscript_utils.h5_lock, h5py
import labscript_utils.properties


class NI_connect(IntermediateDevice):
    allowed_children = [AnalogOut]

    n_analogs = 8
    digital_dtype = np.uint32
    clock_limit = 1e6#500e3

    description = 'NI_PCIe_6713 with ExtClock'

    @set_passed_properties(property_names = {
        "device_properties":["acquisition_rate", "MAX_name"]}
        )

    def __init__(self, name, parent_device=None,clock_terminal=None, MAX_name=None, port=1028, aquisition_rate=0, **kwargs):
        IntermediateDevice.__init__(self, name, parent_device,)
        self.aquisition_rate = aquisition_rate
        self.clock_terminal = clock_terminal
        self.port = port
        self.MAX_name = name if MAX_name is None else MAX_name
        self.BLACS_connection = self.MAX_name+"@"+str(self.port)


    def convert_bools_to_bytes(self,digitals):
        """converts digital outputs to an array of bitfields stored
        as self.digital_dtype"""
        outputarray = [0]*32#da uint32 bits #[0]*self.n_digitals
        for output in digitals:
            #connection is the digital-output Name: do_1, etc
            #raw_output is the hi/low setting for this output for every tick
            port = output.connection.replace('do_','')
            port = int(port) - 1
            outputarray[port] = output.raw_output
            #raw_output: [0 0 0 1 0 ...] value for this output at each clock tick

        #convert bits to uint32: do_1=hi, do_2=hi => 11->3
        #                        do_1=lo, do_2=hi => 01->2
        bits = bitfield(outputarray,dtype=self.digital_dtype)
        #bits = [0 0 1 0 3 ...] combined outputs at each clock tick
        #bits = self.shrink_bitfield(bits)
        return bits

    def generate_code(self, hdf5_file):
        IntermediateDevice.generate_code(self, hdf5_file)

        analogs = {}
        for device in self.child_devices:
            if isinstance(device,AnalogOut):
                analogs[device.connection] = device
#            elif isinstance(device,DigitalOut):
                #emit warning: Digital Outputs cannot be used in buffered Shots!
#                digitals[device.connection] = device
            else:
                raise Exception('Got unexpected device.')
        clockline = self.parent_device
        pseudoclock = clockline.parent_device
        times = pseudoclock.times[clockline]

        analog_out_table = np.empty((len(times),len(analogs)), dtype=np.float32)
        analog_connections = analogs.keys()
        analog_connections.sort()
        analog_out_attrs = []
        for i, connection in enumerate(analog_connections):
            output = analogs[connection]
            if any(output.raw_output > 10 )  or any(output.raw_output < -10 ):
                # Bounds checking:
                raise LabscriptError('%s %s '%(output.description, output.name) +
                                  'can only have values between -10 and 10 Volts, ' +
                                  'the limit imposed by %s.'%self.name)
            analog_out_table[:,i] = output.raw_output
            analog_out_attrs.append(self.MAX_name +'/'+connection)

        analog_out_table_hash = hashlib.md5(analog_out_table).hexdigest()
        grp = self.init_device_group(hdf5_file)
        if all(analog_out_table.shape) and analog_out_table.shape[0]>2: # Both dimensions must be nonzero
            grp.create_dataset('ANALOG_OUTS',compression=config.compression,data=analog_out_table)
            self.set_property('data_hash', analog_out_table_hash, location='device_properties')
            self.set_property('analog_out_channels', ', '.join(analog_out_attrs), location='device_properties')

        self.set_property('clock_terminal', self.clock_terminal, location='connection_table_properties')


import os
from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED

from blacs.device_base_class import DeviceTab

from qtutils import UiLoader
import socket

@BLACS_tab
class NI_ConnectTab(DeviceTab):

    port = 0

    def initialise_GUI(self):

        layout = self.get_tab_layout() #TODO: put above all other control buttons
        ui_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ni_connect.ui')
        self.ui = UiLoader().load(ui_filepath)
        layout.addWidget(self.ui)

        #Capabilities
        self.num_DO = 8
        self.num_AO = 8
        self.ao_base_units     = 'V'
        self.ao_base_min       = -10.0
        self.ao_base_max       =  10.0
        self.ao_base_step      =  0.001
        self.ao_base_decimals  =  3

        analog_properties = {}
        for i in range(self.num_AO):
            analog_properties['ao%d'%i] = {'base_unit':self.ao_base_units,
                                            'min':self.ao_base_min,
                                            'max':self.ao_base_max,
                                            'step':self.ao_base_step,
                                            'decimals':self.ao_base_decimals
                                           }

        digital_properties = {}
        for i in range(self.num_DO):
            digital_properties['do_%d'%i] = {}


        self.create_digital_outputs(digital_properties)
        self.create_analog_outputs(analog_properties)

        dds_widgets,ao_widgets,do_widgets = self.auto_create_widgets()

        self.auto_place_widgets(do_widgets, ao_widgets)

        #self.port = int(self.settings['connection_table'].find_by_name(self.settings["device_name"]).BLACS_connection)
        conn = self.settings['connection_table'].find_by_name(self.settings['device_name']).BLACS_connection
        max_name, port = conn.split('@')
        self.port = int(port)
        self.max_name = max_name

        self.ui.label_port.setText(str(self.port))
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        self.ui.label_host.setText(local_ip)
        self.ui.label_connected.setVisible(False)
        self.ui.label_connected_symbol.setVisible(False)
        #self.ui.label_ok.setVisible(False)
        self.statemachine_timeout_add(2000, self.status_monitor)

#    def get_save_data(self):
#        return {'host': str(self.ui.host_lineEdit.text()), 'use_zmq': self.ui.use_zmq_checkBox.isChecked()}
#
#    def restore_save_data(self, save_data):
#        print 'restore save data running'
#        if save_data:
#            host = save_data['host']
#            self.ui.host_lineEdit.setText(host)
#            if 'use_zmq' in save_data:
#                use_zmq = save_data['use_zmq']
#                self.ui.use_zmq_checkBox.setChecked(use_zmq)
#        else:
#            self.logger.warning('No previous front panel state to restore')
#
#        # call update_settings if primary_worker is set
#        # this will be true if you load a front panel from the file menu after the tab has started
#        if self.primary_worker:
#            self.update_settings_and_check_connectivity()

    def initialise_workers(self):
        worker_initialisation_kwargs = {'max_name':self.max_name, 'port': self.port}
        self.create_worker("main_worker", NI_connectWorker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"
#        self.update_settings_and_check_connectivity()

    @define_state(MODE_MANUAL, True)
    def status_monitor(self,notify_queue=None):
        connected, status = yield(self.queue_work(self._primary_worker,'check_status'))
        #self.ui.label_status.setText(str(status))

        if connected:
            self.ui.label_not_connected.setVisible(False)
            self.ui.label_not_connected_symbol.setVisible(False)
            self.ui.label_connected.setVisible(True)
            self.ui.label_connected_symbol.setVisible(True)
        else:
            self.ui.label_not_connected.setVisible(True)
            self.ui.label_not_connected_symbol.setVisible(True)
            self.ui.label_connected.setVisible(False)
            self.ui.label_connected_symbol.setVisible(False)

        #self.ui.label_ok.setVisible(connected)
        #self.ui.label_error.setVisible(not connected)

    @define_state(MODE_MANUAL,True)
    def transition_to_buffered(self,h5_file,notify_queue):
        DeviceTab.transition_to_buffered(self, h5_file, notify_queue)
        self.statemachine_timeout_remove(self.status_monitor) #remove status check in buffered mode (remove pinging)

    @define_state(MODE_BUFFERED,False)
    def transition_to_manual(self,notify_queue,program=False):
        DeviceTab.transition_to_manual(self, notify_queue, program)
        self.statemachine_timeout_add(2000,self.status_monitor) #re-add status check in manual mode


class NI_connectWorker(Worker):
    def init(self):#, port, host, use_zmq):
#        self.port = port
#        self.host = host
#        self.use_zmq = use_zmq
        global socket; import socket
        global threading; import threading
#        global zmq; import zmq
#        global zprocess; import zprocess
        global h5py; import labscript_utils.h5_lock, h5py
        global logging; import logging
        global shared_drive; import labscript_utils.shared_drive as shared_drive
        global time; import time
        global struct; import struct
        global math; import math

        self.host = ''
        self.use_zmq = False
        self.server_thread = None
        self.server_socket = None

        self.shutdown_server = False

        self.client_connection_socket = None
        self.client_connection_address = ""
        self.BUFFER_SIZE = 16 # Normally 1024, but we want fast response
        self.init_server()

        self.hdf5_filename = None
        self.logger = logging.getLogger('BLACS')

        self.len_packer = struct.Struct('>i')
        self.type_packer = struct.Struct('>h')
        #self.double_packer = struct.Struct('>d')

        self.last_front_panel_values = None
        self.last_data_hash = None

        self.logger = logging.getLogger('BLACS')

#    def update_settings_and_check_connectivity(self, host, use_zmq):
#        self.host = host
#        self.use_zmq = use_zmq
#        if not self.host:
#            return False
#        if not self.use_zmq:
#            return self.initialise_sockets(self.host, self.port)
#        else:
#            response = zprocess.zmq_get_raw(self.port, self.host, data='hello')
#            if response == 'hello':
#                return True
#            else:
#                raise Exception('invalid response from server: ' + str(response))
    def send_string(self, msg):
        type = self.type_packer.pack(0) #type 0: raw string message
        send_msg =  msg.encode('utf-8') #type, msg
        packed_len = self.len_packer.pack(len(send_msg))

        self.client_connection_socket.sendall(packed_len + type + send_msg) #send and flush

    def send_ping(self)    :
        type = self.type_packer.pack(1)
        packed_len = self.len_packer.pack(0)
        self.client_connection_socket.sendall(packed_len + type)

    def send_progam_manual(self, front_panel_values):
        type = self.type_packer.pack(2)
        send_msg = str(front_panel_values).encode('utf-8')
        packet_len = self.len_packer.pack(len(send_msg))

        self.client_connection_socket.sendall(packet_len + type + send_msg)

    def send_transition_to_buffered(self, fresh,  clock_terminal, ao_channels, ao_data):
        type = self.type_packer.pack(3)
        send_message = str({'fresh':fresh, 'clock_terminal':clock_terminal, 'ao_channels':ao_channels})
        send_message = send_message.encode('utf-8')
        packet_len = self.len_packer.pack(len(send_message))

        self.client_connection_socket.send(packet_len + type + send_message)
        if fresh:
            shape = ao_data.shape
            send_data = self.len_packer.pack(shape[0]) + self.len_packer.pack(shape[1])
            self.client_connection_socket.send(send_data)

            #ao_data.dtype = np.float64
            #ao_data.reshape(shape[0]*shape[1])
            #data = self.double_packer.pack(*ao_data)
            send_data = ao_data.astype('>f').tostring()
            self.client_connection_socket.sendall(send_data)
            #self.client_connection_socket.flush()

            #reseived = self.client_connection_socket.recv(128)


    def send_transition_to_manual(self, more_reps, abort):
        type = self.type_packer.pack(4)
        send_message = str({'more_reps':more_reps, 'abort':abort})
        send_message = send_message.encode('utf-8')
        packet_len = self.len_packer.pack(len(send_message))

        self.client_connection_socket.send(packet_len + type + send_message)

        #wait for response
        response = self.client_connection_socket.recv(2)
        response, = self.type_packer.unpack(response)
        if response != 5:
            raise Exception("Wrong response: "+str(response))

    def send_conn_close(self):
        type = self.type_packer.pack(8)
        packed_len = self.len_packer.pack(0)
        self.client_connection_socket.sendall(packed_len + type)

    def check_device_name(self):
        type = self.type_packer.pack(7)
        packed_len = self.len_packer.pack(0)
        self.client_connection_socket.sendall(packed_len + type)

        #wait for response
        response = self.client_connection_socket.recv(4)
        packet_length, = self.len_packer.unpack(response)

        response = self.client_connection_socket.recv(packet_length)
        response = response.decode('utf-8')
        #the response contains the MAX_name of the connection
        if response != self.max_name:
            self.send_string("Wrong MAX_name. Expected: "+self.max_name+". Check MAX_name and port.")
            self.send_conn_close()
            self.client_connection_socket.close()
            self.client_connection_socket = None
            return False
        return True


    def check_status(self):
        if self.client_connection_socket:
            try:
                #self.client_connection_socket.send("ping")
                self.send_ping()
            except:
                self.client_connection_socket = None
                #connection lost
                return False, "listening..."
            return True, "connected to "+str(self.client_connection_address)
        else:
            return False, "listening..."

    def accept_loop(self):
        self.logger.info("ANDOR_CCD: started accept thread")
        while not self.shutdown_server:
            if not self.client_connection_socket and self.server_socket:
                self.logger.info("accept_loop")
                self.client_connection_socket, self.client_connection_address = self.server_socket.accept()
                self.client_connection_socket.settimeout(3)
                self.logger.info("accepted: "+str(self.client_connection_address))
                self.last_data_hash = None
                #request Device Connection MAX_name
                if self.check_device_name() and self.last_front_panel_values: #send current front_panel_values
                #TODO TEST!!!!!!!!!!!!!!!!!!!!!!!
                    self.send_progam_manual(self.last_front_panel_values)
#                success, response =  self.send_to_client("hello",True)
#                if not success or response != "hello":
#                    if self.client_connection_socket:
#                        self.client_connection_socket.close()
#                    self.client_connection_socket = None
            else:
                time.sleep(0.5)


    def init_server(self):
        self.shutdown_server = False
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.logger.info("ANDOR_CCD: port="+str(self.port)+" type="+str(type(self.port)))
        self.server_socket.bind(('', self.port))
        self.server_socket.listen(1)

        self.server_thread = threading.Thread(target=self.accept_loop)
        self.server_thread.setDaemon(True) # don't hang on exit
        self.server_thread.start()

    def close_server(self):
        self.logger.info("ANDOR_CCD: Shutdown server")
        self.shutdown_server = True #make the accept_loop stop
        if self.client_connection_socket:
            self.client_connection_socket.close()
            self.client_connection_socket = None
        self.server_socket.close()
        self.server_socket = None


    def send_to_client(self, message, wait_for_response=False, response_timeout=0.5):
        #self.logger.info("ANDOR_CCD: send to client: "+str(message))
        if self.client_connection_socket:
            try:
                self.client_connection_socket.send(str(message))
            except:
                self.client_connection_socket.close()
                self.client_connection_socket = None
                return False, "Cannot send message to client"

            if wait_for_response:
                self.client_connection_socket.settimeout(response_timeout)
                try:
                    response = self.client_connection_socket.recv(self.BUFFER_SIZE)
                except socket.timeout:
                    self.client_connection_socket.close()
                    self.client_connection_socket = None
                    return False, "No respose received in "+str(response_timeout)+"sec"
                if not response:
                    return False,"The client did not responde"
                else:
                    return True,response
            else:
                return True,"ok"
        else:
            return False,"Cannot send message to client. The camera client is not connected"


    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        self.hdf5_filename = h5file

        with h5py.File(h5file,'r') as hdf5_file:
            group = hdf5_file['devices/'][device_name]
            device_properties = labscript_utils.properties.get(hdf5_file, device_name, 'device_properties')
            new_data_hash = device_properties['data_hash']
            if self.last_data_hash == new_data_hash:
                self.send_transition_to_buffered(False, None, None, None)
                #wait for response
                response = self.client_connection_socket.recv(2)
                response, = self.type_packer.unpack(response)
                if response != 5:
                    raise Exception("Wrong response: "+str(response))
                return {}

            connection_table_properties = labscript_utils.properties.get(hdf5_file, device_name, 'connection_table_properties')
            clock_terminal = connection_table_properties['clock_terminal']
            ao_channels = device_properties['analog_out_channels']

            h5_data = group.get('ANALOG_OUTS')
            if h5_data:
                self.buffered_using_analog = True
                #ao_channels = device_properties['analog_out_channels']
                # We use all but the last sample (which is identical to the
                # second last sample) in order to ensure there is one more
                # clock tick than there are samples. The 6733 requires this
                # to determine that the task has completed.
                ao_data = np.array(h5_data,dtype=np.float32)[:-1,:] #float64?

        self.send_transition_to_buffered(fresh, clock_terminal, ao_channels, ao_data) #fresh==True
        #wait for response
        response = self.client_connection_socket.recv(2)
        response, = self.type_packer.unpack(response)
        if response != 5:
            raise Exception("Wrong response: "+str(response))

        self.last_data_hash = new_data_hash

#        if not self.client_connection_socket:
#            raise Exception("No Camera client is connected")
#
#        h5file = h5file.replace("E:","\\\\APQ-PC")
#        #if live mode -> dont save (send)
#        h5file = shared_drive.path_to_agnostic(h5file)
#
#        success,response = self.send_to_client(h5file, True)
#        if (not success) or (response != 'ok'):
#            raise Exception("no or invalid response from camera client: "+str(response))
#
        return {}

    def transition_to_manual(self, abort=False):
        more_reps = False
        self.send_transition_to_manual(more_reps, abort)
        if abort:
            self.last_data_hash = None
        return True # indicates success

    def abort_transition_to_buffered(self):
        # TODO: untested
        return self.transition_to_manual(True)

    def abort_buffered(self):
        # TODO: untested
        return self.transition_to_manual(True)


#    def abort_buffered(self):
#        return self.abort()
#
#    def abort_transition_to_buffered(self):
#        return self.abort()

    def abort(self):
#        if not self.use_zmq:
#            return self.abort_sockets(self.host, self.port)
#        response = zprocess.zmq_get_raw(self.port, self.host, 'abort')
#        if response != 'done':
#            raise Exception('invalid response from server: ' + str(response))
        return True # indicates success

    def program_manual(self, values):
        self.last_front_panel_values = values
        if self.client_connection_socket:
            self.send_progam_manual(values)
            #send_msg = str(values).encode('utf-8')
            #struct.pack_into('>i', self.client_connection_socket, len(send_msg))
            #self.client_connection_socket.sendall(send_msg) #send and flush
        return {}

    def shutdown(self):
        self.close_server()
#        return


@runviewer_parser
class RunviewerClass(object):
    #num_digitals = 32

    def __init__(self, path, device):
        self.path = path
        self.name = device.name
        self.device = device

        # We create a lookup table for strings to be used later as dictionary keys.
        # This saves having to evaluate '%d'%i many many times, and makes the _add_pulse_program_row_to_traces method
        # significantly more efficient
#        self.port_strings = {}
#        for i in range(self.num_digitals):
#            self.port_strings[i] = 'port0/line%d'%i

    def get_traces(self, add_trace, clock=None):
        if clock is None:
            # we're the master pseudoclock, software triggered. So we don't have to worry about trigger delays, etc
            raise Exception('No clock passed to %s. The NI PCI 6713 must be clocked by another device.'%self.name)

        # get the pulse program
        with h5py.File(self.path, 'r') as f:
            if 'ANALOG_OUTS' in f['devices/%s'%self.name]:
                analogs = f['devices/%s/ANALOG_OUTS'%self.name][:]
                analog_out_channels = labscript_utils.properties.get(f, self.name, 'device_properties')['analog_out_channels'].split(', ')
            else:
                analogs = None
                analog_out_channels = []

        times, clock_value = clock[0], clock[1]

        clock_indices = np.where((clock_value[1:]-clock_value[:-1])==1)[0]+1
        # If initial clock value is 1, then this counts as a rising edge (clock should be 0 before experiment)
        # but this is not picked up by the above code. So we insert it!
        if clock_value[0] == 1:
            clock_indices = np.insert(clock_indices, 0, 0)
        clock_ticks = times[clock_indices]

        traces = {}

        for i, channel in enumerate(analog_out_channels):
            traces[channel.split('/')[-1]] = (clock_ticks, analogs[:,i])

        triggers = {}
        for channel_name, channel in self.device.child_list.items():
            if channel.parent_port in traces:
                if channel.device_class == 'Trigger':
                    triggers[channel_name] = traces[channel.parent_port]
                add_trace(channel_name, traces[channel.parent_port], self.name, channel.parent_port)

        return triggers

