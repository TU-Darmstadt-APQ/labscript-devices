from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED
from blacs.tab_base_classes import MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED
from blacs.device_base_class import DeviceTab
from labscript_devices import runviewer_parser, BLACS_tab
from labscript import bitfield, config, PseudoclockDevice, Pseudoclock, ClockLine, LabscriptError, IntermediateDevice,  DigitalOut, set_passed_properties
import numpy as np
import labscript_utils.h5_lock, h5py
import labscript_utils.properties

import pylab

import time
import logging, logging.handlers

LOGGING = True

if LOGGING:
    logger = logging.getLogger('masterClock')
    handler = logging.handlers.RotatingFileHandler('C:/Temp/MasterClockLogger.log', maxBytes=1024*1024*50)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

def get_port_and_line(connection):
    port, line = connection.replace("port","").replace("line","").split(r"/")
    return int(port), int(line)


class NI_PCIe_6534_Master(PseudoclockDevice):
    allowed_children = [Pseudoclock]

    n_digitals = 32
    digital_dtype = np.uint32

    clock_limit = 1e7 #10MHz
    clock_resolution = 26.6666666666666666e-9

    description = 'NI_PCIe_6534 as Master Clock'

    @set_passed_properties(property_names = {
        "device_properties":["MAX_name"]}
        )

    def __init__(self, name, trigger_device=None, trigger_connection=None, MAX_name=None, set_clock_limit=10e6, external_clock_terminal=None, direct_outputs_sample_clock_output_port='port0/line1', direct_outputs_sample_clock_input_port='PFI3', **kwargs):
        PseudoclockDevice.__init__(self, name, trigger_device, trigger_connection, **kwargs)
        #IntermediateDevice.__init__(self, name, parent_device)


        self.clock_limit = set_clock_limit
        #TODO: differ between clock limit and sample rate -> clock limit is half the sample rate (200us clock at 100us sample)
        self.external_clock_terminal = external_clock_terminal
        self.direct_outputs_sample_clock_input = direct_outputs_sample_clock_input_port
        self.direct_outputs_sample_clock_output = direct_outputs_sample_clock_output_port

        self.MAX_name = name if MAX_name is None else MAX_name
        self.BLACS_connection = self.MAX_name


        #self._internal_clock =  NI_CLOCK('%s_internal_clock'%name, self, sample_rate=self.clock_limit) # possibly a better connection name than 'clock'?
        self._pseudoclock = Pseudoclock('%s_pseudoclock'%name, self, 'clock')
        # Create the internal direct output clock_line
        self._direct_outputs_clock_line = ClockLine('%s_direct_output_clock_line'%name, self._pseudoclock, self.direct_outputs_sample_clock_output, ramping_allowed = False) #since only digital outs are available on this card!
        # Create the internal intermediate device connected to the above clock line
        # This will have the direct DigitalOuts of DDSs of the PulseBlaster connected to it
        self._direct_outputs_device = NI_PCIe_6534_DirectOutputs('%s_direct_output_device'%name, self._direct_outputs_clock_line)

    @property
    def pseudoclock(self):
        return self._pseudoclock

    @property
    def direct_outputs(self):
        return self._direct_outputs_device

    def add_device(self, device):
        #if not self.child_devices and isinstance(device, Pseudoclock):
        if isinstance(device, Pseudoclock):
            PseudoclockDevice.add_device(self, device)

        elif isinstance(device, Pseudoclock):
            raise LabscriptError('The %s %s automatically creates a Pseudoclock because it only supports one. '%(self.description, self.name) +
                                 'Instead of instantiating your own Pseudoclock object, please use the internal' +
                                 ' one stored in %s.pseudoclock'%self.name)
        elif isinstance(device, DigitalOut):
            #TODO: Defensive programming: device.name may not exist!
            raise LabscriptError('You have connected %s directly to %s, which is not allowed. You should instead specify the parent_device of %s as %s.direct_outputs'%(device.name, self.name, device.name, self.name))
        else:
            raise LabscriptError('You have connected %s (class %s) to %s, but %s does not support children with that class.'%(device.name, device.__class__, self.name, self.name))

    def is_con_valid(self, port, line):
        if (-1 < port < 4) and (-1 < line < 8):
            return True
        return False

    def is_con_clock(self, con):
#        for clock_line in self._internal_clock.child_devices:
#            if clock_line.connection == 'internal': #ignore internal clockline
#                continue
#            if con == clock_line.connection:
#                return True
        return False

    def get_direct_outputs(self):
        """Finds out which outputs are directly attached to the PulseBlaster"""
        dig_outputs = []
        for output in self.direct_outputs.get_all_outputs():
            # only check DDS and DigitalOuts (so ignore the children of the DDS)
            if isinstance(output, DigitalOut):
                # get connection number and prefix
                try:
                    port, line = get_port_and_line(output.connection)#output.connection.replace("port","").replace("line").split(r"/")
                    assert (port >= 1 and port <= 3) or (line >= 0 and line <= 7)
                except:
                    raise LabscriptError('%s %s has invalid connection string: \'%s\'. '%(output.description,output.name,str(output.connection)) +
                                         'Format must be \'portX/lineY\' with X an integer less than 4 and Y an integer less than 8')
                # run checks on the connection string to make sure it is valid
                if not self.is_con_valid(port, line):#output.connection):
                    raise LabscriptError('%s is set as connected to port%d/line%d of %s. '%(output.name, port, line, self.name) +
                                         'Output port must be an integer from 1 to 3 and line must be an integer from 0 to 7.')
                # TODO: Most of this should be done in add_device() No?
                if self.is_con_clock(output.connection):
                    raise LabscriptError('%s is set as connected to port%d/line%d of %s. '%(output.name, port, line, self.name) +
                                         'This line is already in use as one of the ClockLines.')

                # Check that the connection string doesn't conflict with another output (that is already appended to the list in this loop)
                for other_output in dig_outputs:
                    if output.connection == other_output.connection:
                        raise LabscriptError('%s and %s are both set as connected to %s of %s.'%(output.name, other_output.name, output.connection, self.name))

                # store a reference to the output
                dig_outputs.append(output)

        return dig_outputs

    def generate_raw_output_from_clock_times(self, times, internal_clockline, extra_ticks=0):
        #int_times = self._internal_clock.times[self._direct_outputs_clock_line]
        #since we have a fixed external sample clockrate 10MHz we can calculate the index of the output array for the switch times

        #int_times = np.linspace(0,self.stop_time,1+self.stop_time*self.clock_limit,endpoint=True)

        #line values should be default 1, go to 0 and back to 1 when generating a clock signal (go back to 1 on time)

        # the hardware requires an instruction amount of a multiple of 4 so coercentasdyfsdgb it
        amount = 1+self.stop_time*self.clock_limit
        reminder = int(amount%4)
#        if reminder == 0:
#            reminder = 4 #always add some instructions
        amount = int(amount + (4-reminder ))
#        div, rem = divmod(amount,4)
#        if rem != 0:
#            amount = amount + rem

        #result = np.ones(1+self.stop_time*self.clock_limit, dtype=np.uint32)
        result = np.ones(amount, dtype=np.uint16)
        #result = np.empty(len(int_times),dtype=np.uint32)#np.empty(len(int_times))

        first_index = -1

        for t in times:
            times_index = int(t*self.clock_limit)
            if first_index == -1:
                first_index = times_index
            result[times_index]   = 0
            result[times_index+1] = 1 #should always be possible because we always add some extra instructions
        if internal_clockline:
            result[first_index + 3] = 0
            result[first_index + 4] = 1
        if extra_ticks > 0:
            if internal_clockline:
                result = np.append(result, np.uint16([0,1])*extra_ticks)
            else:
                result = np.append(result, np.uint16([1,1])*extra_ticks)

        reminder = len(result) % 4
        if reminder > 0:
            result = np.append(result, np.uint16([1])*(4-reminder))

#            if times_index >= len(times)-1:
#                result[times_index-1] = 0
#                result[times_index]   = 1 #last tick
#            else:
#                result[times_index]   = 0
#                result[times_index+1] = 1

#            if(times_index == 0):
##                result[times_index+1] = 0
##                result[times_index+2] = 1
#            #does it work?
#                result[times_index] = 0
#                result[times_index+1] = 1 #causes 100ns later tick...
#            else:
##                result[times_index-1] = 0
##                result[times_index]   = 1
#                result[times_index]   = 0
#                result[times_index+1] = 1 #to compensate 100ns delay from above

#        dt = (int_times[1]-int_times[0])/2.0 # half

#        for i, t in enumerate(int_times):
#            if abs(times[times_index]-t) <= dt:
#                if i == 0:
#                    result[i+1] = 0
#                    result[i+2] = 1
#                else:
#                    result[i-1] = 0
#                    result[i] = 1
#                times_index += 1
#            else:
#                result[i] = 1

#        result[len(int_times)-3]=0 #add tick for end
#        result[len(int_times)-2]=1

        return result

    def convert_to_inst(self, dig_outputs):
        outputarray_clock = [0]*32#8
        outputarray_digi = [0]*32#24
        #clock_ticks = 0
        #outputarray = [0]*32
        #insert clock line ports

        if LOGGING:
            t0 = time.time()
        extra_instructions = 0
        #now insert direct direct_outputs
        if dig_outputs:
            for digi in dig_outputs:
                port, line = get_port_and_line(digi.connection)
                raw_out = digi.raw_output#np.astype(digi.raw_output,np.uint16) #change type from uint32 to uint16, otherwise it will lead to an error in bitfield(...)
                extra_instructions = len(raw_out) % 4
                if extra_instructions > 0:
                    raw_out = np.append(raw_out,[raw_out[-1]]*extra_instructions)
                outputarray_digi[8*(port-2)+line] = raw_out#digi.raw_output #since port 0 and 1 are for clock lines, port 2-3 for direct outputs

#            dummy0 = np.zeros(len(dig_outputs[0].raw_output),dtype=np.uint32)
#            for i, arr in enumerate(outputarray_digi):
#                if isinstance(arr, int):
#                    outputarray_digi[i] = dummy0

            bits_digi = bitfield(outputarray_digi, dtype=np.uint16)
        else:
            bits_digi = None

        if LOGGING:
            t1 = time.time()
            logger.info("ConvToInstr: GenerateDigi Output: "+str(t1-t0))
            t0 = time.time()

        last_clock_line_index=0
        for clock_line in self.pseudoclock.child_devices: #~56ms bei 3 ClockLines (1 interne + 2externe)
            port, line = get_port_and_line(clock_line.connection)
            if port:
                raise LabscriptError('You have connected %s to %s on port %d, which is not allowed (by now). You should connect it to port 0 instead'%(clock_line.name, self.name, port))
            if clock_line in self.pseudoclock.times:
                raw_output = self.generate_raw_output_from_clock_times(self.pseudoclock.times[clock_line], clock_line.name==self._direct_outputs_clock_line.name, extra_instructions)
                #clock_ticks = len(raw_output)
                outputarray_clock[port*8+line] = raw_output
                if(port*8+line)>last_clock_line_index:
                    last_clock_line_index = port*8+line

        if LOGGING:
            t1 = time.time()
            logger.info("ConvToInstr: Generate Raw Outputs from Clock Times: "+str(t1-t0))



#        t0 = time.time()

#        dummy0 = np.zeros(clock_ticks,dtype=np.uint32)
#        for i, arr in enumerate(outputarray_clock):
#            if isinstance(arr,int): #so it is 0, no real output is connected, so fill it with zeros
#                outputarray_clock[i] = dummy0

#        t1 = time.time()
#        self.logger.info("Fill ClockLines with zeros: "+str(t1-t0))
        if LOGGING:
            t0 = time.time()

        bits_clock = bitfield(outputarray_clock, dtype=np.uint16) #~138ms bei 3 Clock Lines
        if LOGGING:
            t1 = time.time()
            logger.info("ConvToInstr: Generate Bitfield: "+str(t1-t0))

        return bits_clock, bits_digi, last_clock_line_index

    def generate_code(self, hdf5_file):
        # Generate the hardware instructions
        if LOGGING:
            total = time.time()
            t0 = time.time()
        hdf5_file.create_group('/devices/'+self.name)
        if LOGGING:
            t1 = time.time()
            logger.info("Create Group: MasterCard: "+str(t1-t0))

        if LOGGING:
            t0 = time.time()
        PseudoclockDevice.generate_code(self, hdf5_file) #~15ms
        if LOGGING:
            t1 = time.time()
            logger.info("Generate Clock: "+str(t1-t0))
        if LOGGING:
            t0 = time.time()
        dig_outputs = self.get_direct_outputs() #~0ms
        if LOGGING:
            t1 = time.time()
            logger.info("Get Direct Outs: "+str(t1-t0))
        if LOGGING:
            t0 = time.time()
        inst_clock, inst_digi, last_clock_line_index = self.convert_to_inst(dig_outputs) #~200ms
        if LOGGING:
            t1 = time.time()
            logger.info("Convert to Instructions: "+str(t1-t0))

            t0 = time.time()
        #SAVE TO FILE: ~386ms
        grp = hdf5_file['/devices/'+self.name]
        if len(inst_clock): # Table must be non empty
            #grp.create_dataset('CLOCK_OUTS',compression=config.compression,data=inst_clock)
            if LOGGING:
                t1 = time.time()
            grp.create_dataset('CLOCK_OUTS',compression="gzip", compression_opts=1,data=inst_clock) #compressen 1/2 is performance optimum
            if LOGGING:
                logger.info("Save clock instr: "+str(time.time()-t1))

            #TODO: just temporary
            self.set_property('last_clock_line_index',last_clock_line_index,location='device_properties')
            self.set_property('clock_lines',  ",".join((self.MAX_name+'/port0/line0:7', self.MAX_name+'/port1/line0:7')), location='device_properties')
        if inst_digi is not None and len(inst_digi):
            grp.create_dataset('DIGITAL_OUTS',compression=config.compression,data=inst_digi)
            self.set_property('digital_lines', ','.join(('/'.join((self.MAX_name,'port2','line0:7')),'/'.join((self.MAX_name,'port3','line0:7')))), location='device_properties')

        #self.set_property('clock_terminal', self.clock_terminal, location='connection_table_properties')
        self.set_property('direct_outputs_sample_clock_input', self.direct_outputs_sample_clock_input, location='connection_table_properties')
        self.set_property('direct_outputs_sample_clock_output', self.direct_outputs_sample_clock_output, location='connection_table_properties')
        self.set_property('clock_limit',self.clock_limit, location='device_properties')
        self.set_property('external_clock_terminal', self.external_clock_terminal, location='connection_table_properties')

#        self.set_property('is_master_pseudoclock', self.is_master_pseudoclock, location='device_properties')
        self.set_property('stop_time', self.stop_time, location='device_properties')
        if LOGGING:
            t1 = time.time()
            logger.info("Save to File: "+str(t1-t0))
            logger.info("Total Master Card: "+str(time.time() - total))


class NI_PCIe_6534_DirectOutputs(IntermediateDevice):
    allowed_children = [DigitalOut]
    clock_limit = NI_PCIe_6534_Master.clock_limit
    description = 'NI_PCIe_6534 Direct Outputs'

    def add_device(self, device):
        port, line = get_port_and_line(device.connection)
        if port >= 2 and port <= 3:
            IntermediateDevice.add_device(self, device)
        else:
            raise LabscriptError("You cannot connect DigitalOut \'%s\' to \'%s\' on port %d. Port must be 2 or 3. Port 0 and 1 are reserved for ClockLines"%(device.name, self.name, port))

import traceback


@runviewer_parser
class RunviewerClass(object):

    def __init__(self, path, device):
        self.path = path
        self.name = device.name
        self.device = device

        self.logger = logging.getLogger('master_viewer_logger')
        handler = logging.handlers.RotatingFileHandler('C:/Temp/MasterViewer.log', maxBytes=1024*1024*50)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        handler.setLevel(logging.DEBUG)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)
        # We create a lookup table for strings to be used later as dictionary keys.
        # This saves having to evaluate '%d'%i many many times, and makes the _add_pulse_program_row_to_traces method
        # significantly more efficient
        self.port_strings = {}
        for port in range(0,4):
            for line in range(0,8):
                self.port_strings[port, line] = 'port%d/line%d'%(port, line)


    def get_traces(self, add_trace, parent=None):# clock=None):
        if parent is None:
            pass
        #if clock is None:
            # we're the master pseudoclock, software triggered. So we don't have to worry about trigger delays, etc
        #    raise Exception('No clock passed to %s. The NI PCIe 6363 must be clocked by another device.'%self.name)
        #raise Exception('test')
        # get the pulse program
        with h5py.File(self.path, 'r') as f:
            group = f['devices/'][self.name]
            device_properties = labscript_utils.properties.get(f, self.name, 'device_properties')
            if 'CLOCK_OUTS' in group:
                h5_data = group.get('CLOCK_OUTS')
                clock_bitfield = np.array(h5_data, dtype=np.uint16)
            else:
                clock_bitfield = []

            if 'DIGITAL_OUTS' in group:
                h5_data = group.get('DIGITAL_OUTS')
                digital_bitfield = np.array(h5_data, dtype=np.uint16)
            else:
                digital_bitfield = []

        port_data = np.zeros((clock_bitfield.shape[0],32),dtype=np.uint8)
#        clock_data = np.zeros((clock_bitfield.shape[0],32), dtype=np.uint8)

        for i in range(0,16):
            port_data[:,i] = (clock_bitfield & (1 << i)) >> i

        if digital_bitfield:
            for i in range(16,32):
                port_data[:,i] = (digital_bitfield & (1 << i)) >> i


        sample_rate = float(device_properties['clock_limit'])
        stop_time = device_properties['stop_time']

        internal_clock_times = np.linspace(0,stop_time,1+stop_time*sample_rate) #since it is a constant ticking sampling clock

        traces = {}
        for port in range(2,4):
            for line in range(0,8):
                traces[self.port_strings[port,line]] = []

        for i in range(0,32):
            traces[self.port_strings[divmod(i,8)]] = (internal_clock_times, port_data[:,i])

        clock_line_times = {}
        for i in range(0,16):
            if any(port_data[:,i]): #disabled clocklines have only zeros. Skip them
                clock_line_times[i] = np.where(port_data[:,i]==0)[0]/sample_rate

        clocklines_and_triggers = {}
        for pseudoclock_name, pseudoclock in self.device.child_list.items():
#            self.logger.info("pseudodlock: "+pseudoclock_name)
            for clock_line_name, clock_line in pseudoclock.child_list.items():
#                self.logger.info("clock_line: "+clock_line_name+" port="+str(clock_line.parent_port))
                if clock_line.parent_port == 'internal':
                    #parent_device_name = '%s.direct_outputs'%self.name
                    parent_device_name = '%s_direct_output_device'%self.name
                    for internal_device_name, internal_device in clock_line.child_list.items():
                        self.logger.info("int Dev Name: "+internal_device_name)
                        if internal_device_name == parent_device_name:
                            for channel_name, channel in internal_device.child_list.items():
                                self.logger.info("ch name: "+channel_name)
                                add_trace(channel_name, traces[channel.parent_port], parent_device_name, channel.parent_port)
#                        else:
#                            clocklines_and_triggers[internal_device_name] = add_trace
#                            for channel_name, channel in internal_device.child_list.items():
#                                clocklines_and_triggers[]
#                                self.logger.info("ch name: "+channel_name)


#                            if channel.device_class == 'Trigger':
#                                clocklines_and_triggers[channel_name] = to_return[channel.parent_port]
#                                add_trace(channel_name, to_return[channel.parent_port], parent_device_name, channel.parent_port)
#                            else:
#                                if channel.device_class == 'DDS':
#                                    for subchnl_name, subchnl in channel.child_list.items():
#                                        connection = '%s_%s'%(channel.parent_port, subchnl.parent_port)
#                                        if connection in to_return:
#                                            add_trace(subchnl.name, to_return[connection], parent_device_name, connection)
#                                else:
#                                    add_trace(channel_name, to_return[channel.parent_port], parent_device_name, channel.parent_port)
                else:
                    try:
                        port, line = get_port_and_line(clock_line.parent_port)
                        if(port*8+line in clock_line_times):
                            clocklines_and_triggers[clock_line_name] = clock_line_times[port*8+line]
                    #add clocking signal to traces, optional, for now dont add a trce for them
                    #add_trace(clock_line_name, traces[clock_line.parent_port], self.name, clock_line.parent_port)
                        #self.logger.info("trace ok")
#                    clocklines_and_triggers[clock_line_name] = to_return[clock_line.parent_port]
#                    add_trace(clock_line_name, to_return[clock_line.parent_port], self.name, clock_line.parent_port)
                    except Exception:
                        self.logger.info(str(traceback.format_exc()))


        return clocklines_and_triggers

@BLACS_tab
class NI_PCIe_6534_MasterDeviceTab(DeviceTab):

    def initialise_GUI(self):
        self.logger.info("Device: init GUI")
        #Capabilities
        #self.num_DO = 32

        digital_properties = {}
        #for i in range(self.num_DO):
        for port in range(0,4):
            for line in range(0,8):
                digital_properties['port%d/line%d'%(port,line)] = {}


        self.create_digital_outputs(digital_properties)

        dds_widgets,ao_widgets,do_widgets = self.auto_create_widgets()

        self.auto_place_widgets(do_widgets)
        #add/append custom widgets possible
        #self.get_tab_layout()

        self.supports_remote_value_check(False)
        self.supports_smart_programming(False)

#        self.statemachine_timeout_add(1000, self.status_monitor)

    @define_state(MODE_MANUAL|MODE_BUFFERED|MODE_TRANSITION_TO_BUFFERED|MODE_TRANSITION_TO_MANUAL,True)
    def status_monitor(self,notify_queue=None):
        # When called with a queue, this function writes to the queue
        # when the pulseblaster is waiting. This indicates the end of
        # an experimental run.
        self.status = yield(self.queue_work(self._primary_worker,'check_status'))

        if notify_queue is not None and self.status == 'finished':#done_condition and not waits_pending:
            # Experiment is over. Tell the queue manager about it, then
            # set the status checking timeout back to every 2 seconds
            # with no queue.
            notify_queue.put('done')
            self.statemachine_timeout_remove(self.status_monitor)
#            self.statemachine_timeout_add(1000,self.status_monitor)

    def initialise_workers(self):
        self.logger.info("Device: init workers")
        self.MAX_name = str(self.settings['connection_table'].find_by_name(self.device_name).BLACS_connection)
        self.create_worker('NI_PCIe_6534_Master_worker', NI_PCIe_6534_Master_Worker, {'MAX_name':self.MAX_name})

        self.primary_worker = "NI_PCIe_6534_Master_worker"

    @define_state(MODE_MANUAL|MODE_BUFFERED|MODE_TRANSITION_TO_BUFFERED|MODE_TRANSITION_TO_MANUAL,True)
    def start(self,widget=None):
        yield(self.queue_work(self._primary_worker,'start_run'))
#        self.status_monitor()

    @define_state(MODE_MANUAL|MODE_BUFFERED|MODE_TRANSITION_TO_BUFFERED|MODE_TRANSITION_TO_MANUAL,True)
    def stop(self,widget=None):
        yield(self.queue_work(self._primary_worker,'transition_to_manual'))
        self.status_monitor()

    @define_state(MODE_MANUAL|MODE_BUFFERED|MODE_TRANSITION_TO_BUFFERED|MODE_TRANSITION_TO_MANUAL,True)
    def reset(self,widget=None):
        yield(self.queue_work(self._primary_worker,'transition_to_manual'))
        self.status_monitor()

    @define_state(MODE_BUFFERED,True)
    def start_run(self, notify_queue):
        """Starts the Pulseblaster, notifying the queue manager when
        the run is over"""
#        self.statemachine_timeout_remove(self.status_monitor)
        self.start()
        self.statemachine_timeout_add(50,self.status_monitor,notify_queue)


class NI_PCIe_6534_Master_Worker(Worker):
    def init(self):
        exec('from PyDAQmx import Task', globals())
        exec('from PyDAQmx.DAQmxConstants import *', globals())
        exec('from PyDAQmx.DAQmxTypes import *', globals())
        global pylab; import pylab
        global h5py; import labscript_utils.h5_lock, h5py
        global numpy; import numpy
        global time; import time


        # Create DO task:
        self.do_task = Task()
        self.clock_line_task = Task()

        self.do_read = int32()
        self.do_data = numpy.zeros(32, dtype=numpy.uint8)

        self.setup_static_channels()

        #DAQmx Start Code
        #self.do_task.StartTask()
        self.clock_line_task.StartTask()

        self.logger = logging.getLogger('BLACS')
#        self.logger.info("Worker: init")

#        self.rerun = False
#        self.oldh5name = None
#        self.buffer_old_clock = None
#        self.buffer_old_do    = None

        self.run_task_again = False
        self.h5file = ""


    def setup_static_channels(self):
        #setup DO ports
        #self.do_task.CreateDOChan(self.MAX_name+"/port0/line0:7,"+self.MAX_name+"/port1/line0:7,"+self.MAX_name+"/port2/line0:7,"+self.MAX_name+"/port3/line0:7","",DAQmx_Val_ChanForAllLines)
        self.clock_line_task.CreateDOChan(self.MAX_name+"/port0/line0:7,"+self.MAX_name+"/port1/line0:7,"+self.MAX_name+"/port2/line0:7,"+self.MAX_name+"/port3/line0:7","",DAQmx_Val_ChanForAllLines)


    def shutdown(self):
        #self.do_task.StopTask()
        #self.do_task.ClearTask()
        self.clock_line_task.StopTask()
        self.clock_line_task.ClearTask()

    def start_run(self):
        t0 = time.time()
        self.clock_line_task.StartTask() #~230ms
        self.logger.info("MASTER: START TASK: "+str(time.time()-t0))

    def check_status(self):
        clock_done = c_ulong()
        self.clock_line_task.IsTaskDone(byref(clock_done))
        return 'finished' if clock_done else 'running'

    def program_manual(self,front_panel_values):
        for port in range(0,4):
            for line in range(0,8):
                self.do_data[port*8+line] = front_panel_values["port%d/line%d"%(port,line)]

        self.clock_line_task.WriteDigitalLines(1,True,1,DAQmx_Val_GroupByChannel,self.do_data,byref(self.do_read),None)
        # TODO: Return coerced/quantised values
        return {}

    def transition_to_buffered(self,device_name,h5file,initial_values,fresh):
        if LOGGING:
            t0 = time.time()
        # Store the initial values in case we have to abort and restore them:
        # TODO: Coerce/quantise these correctly before returning them
        self.initial_values = initial_values
        self.h5file = h5file

        with h5py.File(h5file,'r') as hdf5_file:
            group = hdf5_file['devices/'][device_name]
            attribs = hdf5_file.attrs;
            device_properties = labscript_utils.properties.get(hdf5_file, device_name, 'device_properties')
            connection_table_properties = labscript_utils.properties.get(hdf5_file, device_name, 'connection_table_properties')
            last_clock_line_index = device_properties["last_clock_line_index"]

            h5_data = group.get('DIGITAL_OUTS')
            if h5_data:
                self.buffered_using_digital = True
                do_channels = device_properties['digital_lines']
                do_bitfield = numpy.array(h5_data,dtype=numpy.uint16)
            else:
                self.buffered_using_digital = False

            h5_data = group.get('CLOCK_OUTS')
            if h5_data:
                self.buffered_using_clock = True
                clock_channels = device_properties['clock_lines']
                clock_bitfield = numpy.array(h5_data,dtype=numpy.uint16)
            else:
                self.buffered_using_clock = False

        final_values = {}
        if self.buffered_using_clock:
            clock_write_data = numpy.zeros((clock_bitfield.shape[0],16),dtype=numpy.uint8)
            for i in range(0,last_clock_line_index+1): #only parse neccessary lines!
                clock_write_data[:,i] = (clock_bitfield & (1 << i)) >> i

            self.clock_line_task.StopTask()
            self.clock_line_task.ClearTask()

            self.clock_line_task = Task()
            self.do_read = int32()

            self.clock_line_task.CreateDOChan(clock_channels, "", DAQmx_Val_ChanPerLine) #r"Dev3/port0/line0:7"
            #self.clock_line_task.CfgSampClkTiming(connection_table_properties['external_clock_terminal'], 1e7, DAQmx_Val_Rising,DAQmx_Val_FiniteSamps, clock_bitfield.shape[0])#PFI2
            self.clock_line_task.CfgSampClkTiming(connection_table_properties['external_clock_terminal'], device_properties['clock_limit'], DAQmx_Val_Rising,DAQmx_Val_FiniteSamps, clock_write_data.shape[0])#PFI2
            self.clock_line_task.WriteDigitalLines(clock_write_data.shape[0],False,10.0,DAQmx_Val_GroupByScanNumber,clock_write_data,self.do_read,None)

            for port in range(0,2):
                for line in range(0,8):
                    final_values['port%d/line%d'%(port,line)] = clock_write_data[-1,port*8+line]
        else:
            self.clock_line_task.StopTask()
            self.clock_line_task.ClearTask()

        if self.buffered_using_digital:
            do_write_data = numpy.zeros((do_bitfield.shape[0],32),dtype=numpy.uint8)
            for i in range(0,16):#self.num_DO):
                do_write_data[:,i] = (do_bitfield & (1 << i)) >> i #i-te spalte von 32-Ausgaengen

            self.do_task = Task()
            self.do_read = int32()

            self.do_task.CreateDOChan(do_channels, "", DAQmx_Val_ChanPerLine) #r"Dev3/port1/line0:7,Dev3/port2/line0:7,Dev3/port3/line0:7"
            self.do_task.CfgSampClkTiming(connection_table_properties['direct_outputs_sample_clock_input'], 1e7, DAQmx_Val_Rising,DAQmx_Val_FiniteSamps, do_write_data.shape[0])
            self.do_task.WriteDigitalLines(do_write_data.shape[0],False,10.0,DAQmx_Val_GroupByScanNumber,do_write_data,self.do_read,None)

            self.do_task.StartTask()

            for port in range(2,4):
                for line in range(0,8):
                    final_values['port%d/line%d'%(port,line)] = do_write_data[-1,8*(port-2)+line]

        else:
            # We still have to stop the task to make the
            # clock flag available for buffered analog output, or the wait monitor:
            self.do_task.StopTask()
            # self.do_task.ClearTask()
        if LOGGING:
            self.logger.info("TIMING: trans to buff Master: "+str(time.time()-t0))
        return final_values

    def transition_to_manual(self, abort=False):
        t0 = time.time()

        # if aborting, don't call StopTask since this throws an
        # error if the task hasn't actually finished!

        if self.buffered_using_digital:
            if not abort:
                self.do_task.StopTask()
            self.do_task.ClearTask()

        if self.buffered_using_clock:
            if not abort:
                self.clock_line_task.StopTask()
            self.clock_line_task.ClearTask()

        self.clock_line_task = Task()
        self.setup_static_channels()

        self.clock_line_task.StartTask()

        if abort:
            # Reprogram the initial states:
            self.program_manual(self.initial_values)
        self.logger.info("TIMING_MASTER: Trans to Manual: "+str(time.time()-t0))
        return True

    def abort_transition_to_buffered(self):
        # TODO: untested
        return self.transition_to_manual(True)

    def abort_buffered(self):
        # TODO: untested
        return self.transition_to_manual(True)
