#####################################################################
#                                                                   #
# /NI_DAQmx/blacs_workers.py                                        #
#                                                                   #
# Copyright 2018, Monash University, JQI, Christopher Billington    #
#                                                                   #
# This file is part of the module labscript_devices, in the         #
# labscript suite (see http://labscriptsuite.org), and is           #
# licensed under the Simplified BSD License. See the license.txt    #
# file in the root of the project for the full license.             #
#                                                                   #
#####################################################################
from asyncio import current_task
import sys
import time
import threading
import json
from PyDAQmx import *
from PyDAQmx.DAQmxConstants import *
from PyDAQmx.DAQmxTypes import *
from PyDAQmx.DAQmxCallBack import *

import pandas as pd

import zmq

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import labscript_utils.h5_lock
import h5py
from zprocess import Event
from zprocess.utils import _reraise

import labscript_utils.properties as properties
from labscript_utils import dedent
from labscript_utils.connections import _ensure_str

from blacs.tab_base_classes import Worker

from .utils import split_conn_port, split_conn_DO, split_conn_AI
from .daqmx_utils import incomplete_sample_detection

from labscript_utils.in_exp_com import RunBaseClass
from labscript_utils.labconfig import LabConfig


if sys.platform != 'win32':
    from time import perf_counter
    try:
        from time import perf_counter_ns
    except ImportError:
        def perf_counter():
            """perf_counter_ns() -> int

            Performance counter for benchmarking as nanoseconds.
            """
            return perf_counter()
else:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    kernel32.QueryPerformanceFrequency.argtypes = (
        wintypes.PLARGE_INTEGER,) # lpFrequency

    kernel32.QueryPerformanceCounter.argtypes = (
        wintypes.PLARGE_INTEGER,) # lpPerformanceCount

    _qpc_frequency = wintypes.LARGE_INTEGER()
    if not kernel32.QueryPerformanceFrequency(ctypes.byref(_qpc_frequency)):
        raise ctypes.WinError(ctypes.get_last_error())
    _qpc_frequency = _qpc_frequency.value

    def perf_counter():
        """perf_counter() -> float

        Performance counter for benchmarking.
        """
        count = wintypes.LARGE_INTEGER()
        if not kernel32.QueryPerformanceCounter(ctypes.byref(count)):
            raise ctypes.WinError(ctypes.get_last_error())
        return count.value / _qpc_frequency



class NI_DAQmxOutputWorker(Worker):
    def init(self):


        self.check_version()
        # Reset Device: clears previously added routes etc. Note: is insufficient for
        # some devices, which require power cycling to truly reset.
        DAQmxResetDevice(self.MAX_name)
        self.start_manual_mode_tasks()
        
        self.runner = RunBaseClass(self.name, self.jump_address)
        self.runner.start()

        self.DO_active = False
        self.AO_active = False

        self.current_section = 0

        self.sections = []

        self.AO_all_zero = True
        self.DO_all_zero = True

        def is_finished_callback():
            WAIT_TIME = 0.5
            if self.AO_task is not None and self.AO_active:
                try:
                    self.AO_task.WaitUntilTaskDone(WAIT_TIME)
                except:
                    return False
                self.AO_task.StopTask()

                
            if self.DO_task is not None and self.DO_active:
                try:
                    self.DO_task.WaitUntilTaskDone(WAIT_TIME)
                except:
                    return False
                self.DO_task.StopTask()

            return True

        def load_section(next_section):
            start_t = time.perf_counter()
            if self.current_section != next_section:
                if not self.static_AO and not self.AO_all_zero and self.AO_active:
                    self.AO_task.ClearTask()
                if not self.static_DO and not self.DO_all_zero and self.DO_active:
                    self.DO_task.ClearTask()
                self.program_buffered_AO(self.sections[next_section]['AO_values'])
                self.program_buffered_DO(self.sections[next_section]['DO_values'])
            else:
                if self.AO_task is not None and self.AO_active:
                    self.AO_task.StartTask()
                if self.DO_task is not None and self.DO_active:
                    self.DO_task.StartTask()
            req_t = time.perf_counter() - start_t
            self.programming_times.append(req_t)

        self.runner.set_is_finished_callback(is_finished_callback)
        self.runner.set_load_next_section_callback(load_section)

        self.programming_times = []
        self.h5_file = None

    def stop_tasks(self):
        if self.AO_task is not None:
            self.AO_task.StopTask()
            self.AO_task.ClearTask()
            self.AO_task = None
        if self.DO_task is not None:
            self.DO_task.StopTask()
            self.DO_task.ClearTask()
            self.DO_task = None

    def shutdown(self):
        self.runner.shutdown()
        self.stop_tasks()
        

    def check_version(self):
        """Check the version of PyDAQmx is high enough to avoid a known bug"""
        major = uInt32()
        minor = uInt32()
        patch = uInt32()
        DAQmxGetSysNIDAQMajorVersion(major)
        DAQmxGetSysNIDAQMinorVersion(minor)
        DAQmxGetSysNIDAQUpdateVersion(patch)

        if major.value == 14 and minor.value < 2:
            msg = """There is a known bug with buffered shots using NI DAQmx v14.0.0.
                This bug does not exist on v14.2.0. You are currently using v%d.%d.%d.
                Please ensure you upgrade to v14.2.0 or higher."""
            raise Exception(dedent(msg) % (major.value, minor.value, patch.value))

    def start_manual_mode_tasks(self):
        # Create tasks:
        if self.num_AO > 0:
            self.AO_task = Task()
        else:
            self.AO_task = None

        if self.ports:
            self.DO_task = Task()
        else:
            self.DO_task = None

        # Setup AO channels
        for i in range(self.num_AO):
            con = self.MAX_name + "/ao%d" % i
            self.AO_task.CreateAOVoltageChan(
                con, "", self.Vmin, self.Vmax, DAQmx_Val_Volts, None
            )

        # Setup DO channels
        for port_str in sorted(self.ports, key=split_conn_port):
            if not self.ports[port_str]['num_lines']:
                continue
            # Add each port to the task:
            con = '%s/%s' % (self.MAX_name, port_str)
            self.DO_task.CreateDOChan(con, "", DAQmx_Val_ChanForAllLines)

        # Start tasks:
        if self.AO_task is not None:
            self.AO_active = True
            self.AO_task.StartTask()
        if self.DO_task is not None:
            self.DO_active = True
            self.DO_task.StartTask()

    def program_manual(self, front_panel_values):
        written = int32()
        if self.AO_task is not None:
            AO_data = np.zeros(self.num_AO, dtype=np.float64)
            for i in range(self.num_AO):
                AO_data[i] = front_panel_values['ao%d' % i]
            self.AO_task.WriteAnalogF64(
                1, True, 1, DAQmx_Val_GroupByChannel, AO_data, written, None
            )
        if self.DO_task is not None:
            # Due to two bugs in DAQmx, we will always pack our data into a uint32 and
            # write using WriteDigitalU32. The first bug is some kind of use of
            # uninitialised memory when using WriteDigitalLines, discussed here:
            # https://bitbucket.org/labscript_suite
            #     /labscript_devices/pull-requests/56/#comment-83671312
            # The second is that using a smaller int dtype sometimes fails even though
            # it is the correct int size for the size of the port. Using a 32 bit int
            # always works, the additional bits are ignored. This is discussed here:
            # https://forums.ni.com/t5/Multifunction-DAQ
            #     /problem-with-correlated-DIO-on-USB-6341/td-p/3344066
            DO_data = np.zeros(len(self.ports), dtype=np.uint32)
            for conn, value in front_panel_values.items():
                if conn.startswith('port'):
                    port, line = split_conn_DO(conn)
                    DO_data[port] |= value << line
            self.DO_task.WriteDigitalU32(
                1, True, 10.0, DAQmx_Val_GroupByChannel, DO_data, written, None
            )
        # TODO: return coerced/quantised values
        return {}

    def get_output_tables(self, h5file, device_name):
        """Return the AO and DO tables rom the file, or None if they do not exist."""
        with h5py.File(h5file, 'r') as hdf5_file:
            group = hdf5_file['devices'][device_name]
            try:
                AO_table = group['AO'][:]
            except KeyError:
                AO_table = None
            try:
                DO_table = group['DO'][:]
            except KeyError:
                DO_table = None
            try:
                times_table = group['TIMES'][:]
            except KeyError:
                times_table = None
            self.do_channels = hdf5_file['devices'][device_name].attrs['digital_lines']
        return AO_table, DO_table, times_table
            

    def set_mirror_clock_terminal_connected(self, connected):
        """Mirror the clock terminal on another terminal to allow daisy chaining of the
        clock line to other devices, if applicable"""
        if self.clock_mirror_terminal is None:
            return
        if connected:
            DAQmxConnectTerms(
                self.clock_terminal,
                self.clock_mirror_terminal,
                DAQmx_Val_DoNotInvertPolarity,
            )
        else:
            DAQmxDisconnectTerms(self.clock_terminal, self.clock_mirror_terminal)

    def program_buffered_DO(self, DO_table):
        """Create the DO task and program in the DO table for a shot. Return a
        dictionary of the final values of each channel in use"""
        print(f"Write DO")
        if DO_table is None or len(DO_table) == 0:
            self.DO_active = False
            return {}
        self.DO_active = True

        # Expand each bitfield int into self.num['num_DO']
        # individual ones and zeros:

        num_DO = 0
        port_offset = {}
        for port_name in self.ports:
                port = self.ports[port_name]
                if port['supports_buffered']:
                    port_offset[port_name] = num_DO
                    num_DO += port['num_lines']

        do_write_data = np.zeros((DO_table.shape[0], num_DO),dtype=np.uint8)
        for i in range(num_DO):
            do_write_data[:,i] = (DO_table & (1 << i)) >> i

        # if self.DO_task is not None:
        #     self.DO_task.StopTask()
        #     self.DO_task.ClearTask()
        self.DO_task = Task()
        do_read = int32()
        self.DO_task.CreateDOChan(self.do_channels,"",DAQmx_Val_ChanPerLine)
        
        self.DO_all_zero = not np.any(DO_table)
        if self.DO_all_zero:
            DO_table = DO_table[0:1]
        if self.static_DO or self.DO_all_zero:
            self.DO_task.StartTask()
            self.DO_task.WriteDigitalLines(1,True,10.0,DAQmx_Val_GroupByChannel,do_write_data,do_read,None)
        else:
            # We use all but the last sample (which is identical to the
            # second last sample) in order to ensure there is one more
            # clock tick than there are samples. The 6733 requires this
            # to determine that the task has completed.

            do_write_data = do_write_data[:-1,:]
            # Set up timing:
            self.DO_task.CfgSampClkTiming(
                self.clock_terminal,
                self.clock_limit,
                DAQmx_Val_Rising,
                DAQmx_Val_FiniteSamps,
                do_write_data.shape[0],
            )
            
            self.DO_task.WriteDigitalLines(
                            do_write_data.shape[0],
                            False,
                            10.0,
                            DAQmx_Val_GroupByScanNumber,
                            do_write_data,
                            do_read,
                            None)
            self.DO_task.StartTask()

        final_values = {}
        
        k = 0
        for port_name in self.ports:
            port = self.ports[port_name]
            if port['supports_buffered']:
                for j in range(port['num_lines']):
                    final_values[f'{port_name}/line{j}'] = do_write_data[-1,k]
                    k += 1

        return final_values

    def program_buffered_AO(self, AO_table):
        print(f"Write AO")
        if AO_table is None or len(AO_table) == 0:
            self.AO_active = False
            return {}

        self.AO_active = True
        self.AO_task = Task()
        written = int32()
        channels = ', '.join(self.MAX_name + '/' + c for c in AO_table.dtype.names)
        self.AO_task.CreateAOVoltageChan(
            channels, "", self.Vmin, self.Vmax, DAQmx_Val_Volts, None
        )

        # Collect the final values of the analog outs:
        final_values = dict(zip(AO_table.dtype.names, AO_table[-1]))

        # Convert AO table to a regular array and ensure it is C continguous:
        AO_table = np.ascontiguousarray(
            structured_to_unstructured(AO_table, dtype=np.float64)
        )

        # Check if AOs are all zero for the whole shot. If they are this triggers a
        # bug in NI-DAQmx that throws a cryptic error for buffered output. In this
        # case, run it as a non-buffered task.
        self.AO_all_zero = not np.any(AO_table)
        if self.AO_all_zero:
            AO_table = AO_table[0:1]

        if self.static_AO or self.AO_all_zero:
            # Static AO. Start the task and write data, no timing configuration.
            self.AO_task.StartTask()
            self.AO_task.WriteAnalogF64(
                1, False, 10.0, DAQmx_Val_GroupByChannel, AO_table, written, None
            )
        else:
            # We use all but the last sample (which is identical to the second last
            # sample) in order to ensure there is one more clock tick than there are
            # samples. This is required by some devices to determine that the task has
            # completed.
            npts = len(AO_table) - 1

            # Set up timing:
            self.AO_task.CfgSampClkTiming(
                self.clock_terminal,
                self.clock_limit,
                DAQmx_Val_Rising,
                DAQmx_Val_FiniteSamps,
                npts,
            )

            # Write data:
            self.AO_task.WriteAnalogF64(
                npts,
                False,  # autostart
                10.0,  # timeout
                DAQmx_Val_GroupByScanNumber,
                AO_table[:-1],  # All but the last sample as mentioned above
                written,
                None,
            )

            # Go!
            self.AO_task.StartTask()

        return final_values


    def filter_data_by_time(self, time, values, min_time, max_time):

        if time is None:
            return None
        if values is None:
            return None

        low_i = np.searchsorted(time, (min_time), side='left')
        high_i = np.searchsorted(time, (max_time), side='right')

        return values[low_i:high_i]

    def compile_sections(self, h5file, device_name):
        # Get the data to be programmed into the output tasks:
        AO_table, DO_table, times_table = self.get_output_tables(h5file, device_name)

        with h5py.File(h5file, 'r') as hdf5_file:
            jumps = hdf5_file['jumps'][:]
            master_clock = hdf5_file['connection table'].attrs['master_pseudoclock']
            end_time = hdf5_file['devices'][master_clock].attrs['stop_time']
            self.do_channels = hdf5_file['devices'][device_name].attrs['digital_lines']
            self.jump_address = hdf5_file['jumps'].attrs['jump_device_address']
            
        timestamps = []
        for j in range(len(jumps)):
            timestamps.append(jumps[j]["time"])
            timestamps.append(jumps[j]["to_time"])

        timestamps.append(0)
        timestamps.append(end_time) # TODO: final time

        timestamps = sorted(set(timestamps))

        self.sections = []
        DO_final_values = {}
        AO_final_values = {}
        for i in range(len(timestamps)-1):

            start = timestamps[i]
            end = timestamps[i+1]
            
            section = {
                "start": start,
                "end": end,
                "AO_values": self.filter_data_by_time(times_table, AO_table, start, end),
                "DO_values": self.filter_data_by_time(times_table, DO_table, start, end)
            }

            if section['AO_values'] is not None and section['AO_values'] is not []:
                AO_final_values = dict(zip(section['AO_values'].dtype.names, section['AO_values'][-1]))

            if section['DO_values'] is not None and section['DO_values'] is not []:
                # TODO: currently hacky, as we compute this twice. here and when we do program buffered do
                num_DO = 0
                port_offset = {}
                for port_name in self.ports:
                        port = self.ports[port_name]
                        if port['supports_buffered']:
                            port_offset[port_name] = num_DO
                            num_DO += port['num_lines']

                do_write_data = np.zeros((section['DO_values'].shape[0], num_DO),dtype=np.uint8)
                for i in range(num_DO):
                    do_write_data[:,i] = (section['DO_values'] & (1 << i)) >> i

                k = 0
                for port_name in self.ports:
                    port = self.ports[port_name]
                    if port['supports_buffered']:
                        for j in range(port['num_lines']):
                            DO_final_values[f'{port_name}/line{j}'] = do_write_data[-1,k]
                            k += 1

            self.sections.append(section)

        return DO_final_values, AO_final_values


    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):

        print(f"1 {perf_counter()}")

        # Store the initial values in case we have to abort and restore them:
        self.initial_values = initial_values
        self.programming_times = []
        self.h5_file = h5file

        # Stop the manual mode output tasks, if any:
        self.stop_tasks()

        # Mirror the clock terminal, if applicable:
        self.set_mirror_clock_terminal_connected(True)

        print(f"2 {perf_counter()}")
        # Compile all sections:
        DO_final_values, AO_final_values = self.compile_sections(h5file, device_name)

        print(f"3 {perf_counter()}")

        # Start first task

        print(f"AO 1 {perf_counter()}")
        self.program_buffered_AO(self.sections[0]['AO_values'])
        print(f"AO 2 {perf_counter()}")
        self.program_buffered_DO(self.sections[0]['DO_values'])


        print(f"4 {perf_counter()}")

        final_values = {}
        final_values.update(DO_final_values)
        final_values.update(AO_final_values)

        # If we are the wait timeout device, then the final value of the timeout line
        # should be its rearm value:
        if self.wait_timeout_device == self.device_name:
            final_values[self.wait_timeout_connection] = self.wait_timeout_rearm_value

        self.current_section = 0
        self.runner.send_buffered()

        print(f"5 {perf_counter()}")

        return final_values

    def transition_to_manual(self, abort=False):
        # Stop output tasks and call program_manual. Only call StopTask if not aborting.
        # Otherwise results in an error if output was incomplete. If aborting, call
        # ClearTask only.

        npts = uInt64()
        samples = uInt64()
        tasks = []
        if self.AO_task is not None:
            tasks.append([self.AO_task, self.static_AO or self.AO_all_zero, 'AO'])
            self.AO_task = None
        if self.DO_task is not None:
            tasks.append([self.DO_task, self.static_DO or self.DO_all_zero, 'DO'])
            self.DO_task = None

        # append programming times for this device
        with h5py.File(self.h5_file, 'a') as hdf5_file:
            # Sometimes the programming time field persists within the hdf5 file. In this case we want to delete it. This may happen after failed shots.
            if f'/data/programming_time_{self.device_name}' in hdf5_file:
                del hdf5_file[f'/data/programming_time_{self.device_name}']
            hdf5_file.create_dataset(f'/data/programming_time_{self.device_name}', data=self.programming_times)

        for task, static, name in tasks:
            if not abort:
                if not static:
                    try:
                        # Wait for task completion with a 1 second timeout:
                        task.WaitUntilTaskDone(1)
                    finally:
                        # Log where we were up to in sample generation, regardless of
                        # whether the above succeeded:
                        task.GetWriteCurrWritePos(npts)
                        task.GetWriteTotalSampPerChanGenerated(samples)
                        # Detect -1 even though they're supposed to be unsigned ints, -1
                        # seems to indicate the task was not started:
                        current = samples.value if samples.value != 2 ** 64 - 1 else -1
                        total = npts.value if npts.value != 2 ** 64 - 1 else -1
                        msg = 'Stopping %s at sample %d of %d'
                        self.logger.info(msg, name, current, total)
                task.StopTask()
            task.ClearTask()

        # Remove the mirroring of the clock terminal, if applicable:
        self.set_mirror_clock_terminal_connected(False)

        # Set up manual mode tasks again:
        self.start_manual_mode_tasks()
        if abort:
            # Reprogram the initial states:
            self.program_manual(self.initial_values)

        self.runner.check_err()

        return True

    def abort_transition_to_buffered(self):
        self.runner.abort()
        return self.transition_to_manual(True)

    def abort_buffered(self):
        self.runner.abort()
        return self.transition_to_manual(True)


class NI_DAQmxAcquisitionWorker(Worker):
    MAX_READ_INTERVAL = 0.2
    MAX_READ_PTS = 10000

    def init(self):
        # Prevent interference between the read callback and the shutdown code:
        self.tasklock = threading.RLock()

        # Assigned on a per-task basis and cleared afterward:
        self.read_array = None
        self.task = None

        # Assigned on a per-shot basis and cleared afterward:
        self.buffered_mode = False
        self.h5_file = None
        self.acquired_data = None
        self.buffered_rate = None
        self.buffered_chans = None

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        exp_config = LabConfig()
        broker_sub_port = int(exp_config.get('ports', 'BLACS_Broker_Sub'))
        self.socket.connect("tcp://127.0.0.1:%d" % broker_sub_port)

        # Hard coded for now. Perhaps we will add functionality to enable
        # and disable inputs in manual mode, and adjust the rate:
        self.manual_mode_chans = self.AI_chans
        self.manual_mode_rate = 1000

        # An event for knowing when the wait durations are known, so that we may use
        # them to chunk up acquisition data:
        self.wait_durations_analysed = Event('wait_durations_analysed')

        # Start task for manual mode
        self.start_task(self.manual_mode_chans, self.manual_mode_rate)

    def shutdown(self):
        if self.task is not None:
            self.stop_task()
        self.socket.close()
        self.context.term()

    def read(self, task_handle, event_type, num_samples, callback_data=None):
        """Called as a callback by DAQmx while task is running. Also called by us to get
        remaining data just prior to stopping the task. Since the callback runs
        in a separate thread, we need to serialise access to instance variables"""
        samples_read = int32()
        with self.tasklock:
            if self.task is None or task_handle != self.task.taskHandle.value:
                # Task stopped already.
                return 0
            self.task.ReadAnalogF64(
                num_samples,
                -1,
                DAQmx_Val_GroupByScanNumber,
                self.read_array,
                self.read_array.size,
                samples_read,
                None,
            )
            # Select only the data read, and downconvert to 32 bit:
            data = self.read_array[: int(samples_read.value), :].astype(np.float32)
            if self.buffered_mode:
                # Append to the list of acquired data:
                self.acquired_data.append(data)
            else:
                # Send it to the broker
                for i, channel in enumerate(self.manual_mode_chans):
                    self.socket.send_multipart(["{} {}\0".format(self.device_name,channel).encode('utf-8'), np.mean(self.read_array, 0)[i]]) # Only send mean for each channel
        return 0

    def start_task(self, chans, rate):
        """Set up a task that acquires data with a callback every MAX_READ_PTS points or
        MAX_READ_INTERVAL seconds, whichever is faster. NI DAQmx calls callbacks in a
        separate thread, so this method returns, but data acquisition continues until
        stop_task() is called. Data is appended to self.acquired_data if
        self.buffered_mode=True, or (TODO) sent to the [whatever the AI server broker is
        called] if self.buffered_mode=False."""

        if self.task is not None:
            raise RuntimeError('Task already running')

        if chans is None:
            return

        # Get data MAX_READ_PTS points at a time or once every MAX_READ_INTERVAL
        # seconds, whichever is faster:
        num_samples = min(self.MAX_READ_PTS, int(rate * self.MAX_READ_INTERVAL))

        self.read_array = np.zeros((num_samples, len(chans)), dtype=np.float64)
        self.task = Task()

        if self.AI_term == 'RSE':
            term = DAQmx_Val_RSE
        elif self.AI_term == 'NRSE':
            term = DAQmx_Val_NRSE
        elif self.AI_term == 'Diff':
            term = DAQmx_Val_Diff
        elif self.AI_term == 'PseudoDiff':
            term = DAQmx_Val_PseudoDiff

        for chan in chans:
            self.task.CreateAIVoltageChan(
                self.MAX_name + '/' + chan,
                "",
                term,
                self.AI_range[0],
                self.AI_range[1],
                DAQmx_Val_Volts,
                None,
            )

        self.task.CfgSampClkTiming(
            "", rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, num_samples
        )
        if self.buffered_mode:
            self.task.CfgDigEdgeStartTrig(self.clock_terminal, DAQmx_Val_Rising)

        # This must not be garbage collected until the task is:
        self.task.callback_ptr = DAQmxEveryNSamplesEventCallbackPtr(self.read)

        self.task.RegisterEveryNSamplesEvent(
            DAQmx_Val_Acquired_Into_Buffer, num_samples, 0, self.task.callback_ptr, 100
        )

        self.task.StartTask()

    def stop_task(self):
        with self.tasklock:
            if self.task is None:
                raise RuntimeError('Task not running')
            # Read remaining data:
            self.read(self.task, None, -1)
            # Stop the task:
            self.task.StopTask()
            self.task.ClearTask()
            self.task = None
            self.read_array = None

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        self.logger.debug('transition_to_buffered')

        # read channels, acquisition rate, etc from H5 file
        with h5py.File(h5file, 'r') as f:
            group = f['/devices/' + device_name]
            if 'AI' not in group:
                # No acquisition
                return {}
            AI_table = group['AI'][:]
            device_properties = properties.get(f, device_name, 'device_properties')

        chans = [_ensure_str(c) for c in AI_table['connection']]
        # Remove duplicates and sort:
        if chans:
            self.buffered_chans = sorted(set(chans), key=split_conn_AI)
        self.h5_file = h5file
        self.buffered_rate = device_properties['acquisition_rate']
        if device_properties['start_delay_ticks']:
            # delay is defined in sample clock ticks, calculate in sec and save for later
            self.AI_start_delay = self.AI_start_delay_ticks*self.buffered_rate
        self.acquired_data = []
        # Stop the manual mode task and start the buffered mode task:
        self.stop_task()
        self.buffered_mode = True
        self.start_task(self.buffered_chans, self.buffered_rate)
        return {}

    def transition_to_manual(self, abort=False):
        self.logger.debug('transition_to_manual')
        #  If we were doing buffered mode acquisition, stop the buffered mode task and
        # start the manual mode task. We might not have been doing buffered mode
        # acquisition if abort() was called when we are not in buffered mode, or if
        # there were no acuisitions this shot.
        if not self.buffered_mode:
            return True
        if self.buffered_chans is not None:
            self.stop_task()
        self.buffered_mode = False
        self.logger.info('transitioning to manual mode, task stopped')
        self.start_task(self.manual_mode_chans, self.manual_mode_rate)

        if abort:
            self.acquired_data = None
            self.buffered_chans = None
            self.h5_file = None
            self.buffered_rate = None
            return True

        with h5py.File(self.h5_file, 'a') as hdf5_file:
            data_group = hdf5_file['data']
            data_group.create_group(self.device_name)
            waits_in_use = len(hdf5_file['waits']) > 0

        if self.buffered_chans is not None and not self.acquired_data:
            msg = """No data was acquired. Perhaps the acquisition task was not
                triggered to start, is the device connected to a pseudoclock?"""
            raise RuntimeError(dedent(msg))
        # Concatenate our chunks of acquired data and recast them as a structured
        # array with channel names:
        if self.acquired_data:
            start_time = time.time()
            dtypes = [(chan, np.float32) for chan in self.buffered_chans]
            raw_data = np.concatenate(self.acquired_data).view(dtypes)
            raw_data = raw_data.reshape((len(raw_data),))
            self.acquired_data = None
            self.buffered_chans = None
            self.extract_measurements(raw_data, waits_in_use)
            self.h5_file = None
            self.buffered_rate = None
            msg = 'data written, time taken: %ss' % str(time.time() - start_time)
        else:
            msg = 'No acquisitions in this shot.'
        self.logger.info(msg)

        return True

    def extract_measurements(self, raw_data, waits_in_use):
        self.logger.debug('extract_measurements')
        if waits_in_use:
            # There were waits in this shot. We need to wait until the other process has
            # determined their durations before we proceed:
            self.wait_durations_analysed.wait(self.h5_file)

        with h5py.File(self.h5_file, 'a') as hdf5_file:
            if waits_in_use:
                # get the wait start times and durations
                waits = hdf5_file['/data/waits']
                wait_times = waits['time']
                wait_durations = waits['duration']
            try:
                acquisitions = hdf5_file['/devices/' + self.device_name + '/AI']
            except KeyError:
                # No acquisitions!
                return
            try:
                measurements = hdf5_file['/data/traces']
            except KeyError:
                # Group doesn't exist yet, create it:
                measurements = hdf5_file.create_group('/data/traces')

            t0 = self.AI_start_delay
            for connection, label, t_start, t_end, _, _, _ in acquisitions:
                connection = _ensure_str(connection)
                label = _ensure_str(label)
                if waits_in_use:
                    # add durations from all waits that start prior to t_start of
                    # acquisition
                    t_start += wait_durations[(wait_times < t_start)].sum()
                    # compare wait times to t_end to allow for waits during an
                    # acquisition
                    t_end += wait_durations[(wait_times < t_end)].sum()
                i_start = int(np.ceil(self.buffered_rate * (t_start - t0)))
                i_end = int(np.floor(self.buffered_rate * (t_end - t0)))
                # np.ceil does what we want above, but float errors can miss the
                # equality:
                if t0 + (i_start - 1) / self.buffered_rate - t_start > -2e-16:
                    i_start -= 1
                # We want np.floor(x) to yield the largest integer < x (not <=):
                if t_end - t0 - i_end / self.buffered_rate < 2e-16:
                    i_end -= 1
                # IBS: we sometimes find that t_end (with waits) gives a time
                # after the end of acquisition.  The following line
                # will produce return a shorter than expected array if i_end
                # is larger than the length of the array.
                values = raw_data[connection][i_start : i_end + 1]
                i_end = i_start + len(values) - 1 # re-measure i_end

                t_i = t0 + i_start / self.buffered_rate
                t_f = t0 + i_end / self.buffered_rate
                times = np.linspace(t_i, t_f, len(values), endpoint=True)
                dtypes = [('t', np.float64), ('values', np.float32)]
                data = np.empty(len(values), dtype=dtypes)
                data['t'] = times
                data['values'] = values
                measurements.create_dataset(label, data=data)

    def abort_buffered(self):
        return self.transition_to_manual(True)

    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)

    def program_manual(self, values):
        return {}


class NI_DAQmxWaitMonitorWorker(Worker):
    def init(self):

        self.programming_times = []

        self.all_waits_finished = Event('all_waits_finished', type='post')
        self.wait_durations_analysed = Event('wait_durations_analysed', type='post')
        self.wait_completed = Event('wait_completed', type='post')

        # Set on a per-shot basis and cleared afterward:
        self.h5_file = None
        self.CI_task = None
        self.DO_task = None
        self.wait_table = None
        self.semiperiods = None
        self.wait_monitor_thread = None
        self.section_processed_waits = []

        # Saved error in case one occurs in the thread, we can raise it later in
        # transition_to_manual:
        self.wait_monitor_thread_exception = None
        # To trigger early shutdown of the wait monitor thread:
        self.shutting_down = False

        # Does this device have the "incomplete sample detection" feature? This
        # determines whether the first sample on our semiperiod counter input task will
        # be automatically discarded before we see it, or whether we will have to
        # discard it ourselves
        self.incomplete_sample_detection = incomplete_sample_detection(self.MAX_name)

        # Data for timeout triggers:
        if self.timeout_trigger_type == 'rising':
            trigger_value = 1
            rearm_value = 0
        elif self.timeout_trigger_type == 'falling':
            trigger_value = 0
            rearm_value = 1
        else:
            msg = 'timeout_trigger_type  must be "rising" or "falling", not "{}".'
            raise ValueError(msg.format(self.timeout_trigger_type))
        self.timeout_trigger = np.array([trigger_value], dtype=np.uint8)
        self.timeout_rearm = np.array([rearm_value], dtype=np.uint8)


        self.runner = RunBaseClass(self.name, self.jump_address)
        self.runner.start()

        def is_finished_callback():
            if self.wait_table is not None and self.wait_monitor_thread is not None:
                return not self.wait_monitor_thread.is_alive()
            return True

        self.runner.set_is_finished_callback(is_finished_callback)
        self.runner.set_load_next_section_callback(self.run_section)

        self.sections = []

    def shutdown(self):
        self.stop_tasks(True)

        self.from_master_socket.close()
        self.to_master_socket.close()

        self.context.term()

    def read_edges(self, npts, timeout=None):
        """Wait up to the given timeout in seconds for an edge on the wait monitor and
        and return the duration since the previous edge. Return None upon timeout."""
        samples_read = int32()
        # If no timeout, call read repeatedly with a 0.2 second timeout to ensure we
        # don't block indefinitely and can still abort.
        if timeout is None:
            read_timeout = 0.2
        else:
            read_timeout = timeout
        read_array = np.zeros(npts)
        while True:
            if self.shutting_down:
                raise RuntimeError('Stopped before expected number of samples acquired')
            try:
                self.CI_task.ReadCounterF64(
                    npts, read_timeout, read_array, npts, samples_read, None
                )
            except SamplesNotYetAvailableError:
                if timeout is None:
                    continue
                return None
            return read_array

    def wait_monitor(self):
        try:
            # Read edge times from the counter input task, indiciating the times of the
            # pulses that occur at the start of the experiment and after every wait. If a
            # timeout occurs, pulse the timeout output to force a resume of the master
            # pseudoclock. Save the resulting
            self.logger.debug('Wait monitor thread starting')
            with self.kill_lock:
                self.logger.debug('Waiting for start of experiment')
                # Wait for the pulse indicating the start of the experiment:
                if self.incomplete_sample_detection:
                    semiperiods = self.read_edges(1, timeout=None)
                else:
                    semiperiods = self.read_edges(2, timeout=None)
                start_time = time.time()
                self.logger.debug('Experiment started, got edges:' + str(semiperiods))
                # May have been one or two edges, depending on whether the device has
                # incomplete sample detection. We are only interested in the second one
                # anyway, it tells us how long the initial pulse was. Store the pulse width
                # for later, we will use it for making timeout pulses if necessary. Note
                # that the variable current_time is labscript time, so it will be reset
                # after each wait to the time of that wait plus pulse_width.
                current_time = pulse_width = semiperiods[-1]
                self.semiperiods.append(semiperiods[-1])
                # Alright, we're now a short way into the experiment.
                for wait in self.wait_table:
                    # How long until when the next wait should timeout?
                    timeout = wait['time'] + wait['timeout'] - current_time
                    timeout = max(timeout, 0)  # ensure non-negative
                    # Wait that long for the next pulse:
                    self.logger.debug('Waiting for pulse indicating end of wait')
                    semiperiods = self.read_edges(2, timeout)
                    # Did the wait finish of its own accord, or time out?
                    if semiperiods is None:
                        # It timed out. If there is a timeout device, send a trigger to
                        # resume the clock!
                        if self.DO_task is not None:
                            msg = """Wait timed out; retriggering clock with {:.3e} s
                                pulse ({} edge)"""
                            msg = msg.format(pulse_width, self.timeout_trigger_type)
                            self.logger.debug(dedent(msg))
                            self.send_resume_trigger(pulse_width)
                        else:
                            msg = """Specified wait timeout exceeded, but there is no
                                timeout device with which to resume the experiment.
                                Continuing to wait."""
                            self.logger.warning(dedent(msg))
                        # Keep waiting for the clock to resume:
                        self.logger.debug('Waiting for pulse indicating end of wait')
                        semiperiods = self.read_edges(2, timeout=None)
                    # Alright, now we're at the end of the wait.
                    self.semiperiods.extend(semiperiods)
                    self.logger.debug('Wait completed')
                    current_time = wait['time'] + semiperiods[-1]
                    # Inform any interested parties that a wait has completed:
                    postdata = _ensure_str(wait['label'])
                    self.wait_completed.post(self.h5_file, data=postdata)
                # Inform any interested parties that waits have all finished:
                self.logger.debug('All waits finished')
                self.all_waits_finished.post(self.h5_file)
        except Exception:
            self.logger.exception('Exception in wait monitor thread:')
            # Save the exception so it can be raised in transition_to_manual
            self.wait_monitor_thread_exception = sys.exc_info()

        self.section_processed_waits.append(self.calc_wait_times())

    def send_resume_trigger(self, pulse_width):
        written = int32()
        # Trigger:
        self.DO_task.WriteDigitalLines(
            1, True, 1, DAQmx_Val_GroupByChannel, self.timeout_trigger, written, None
        )
        # Wait however long we observed the first pulse of the experiment to be. In
        # practice this is likely to be negligible compared to the other software delays
        # here, but in case it is larger we'd better wait:
        time.sleep(pulse_width)
        # Rearm trigger:
        self.DO_task.WriteDigitalLines(
            1, True, 1, DAQmx_Val_GroupByChannel, self.timeout_rearm, written, None
        )

    def stop_tasks(self, abort):
        self.logger.debug('stop_tasks')
        if self.wait_monitor_thread is not None:
            if abort:
                # This will cause the wait_monitor thread to raise an exception within a
                # short time, allowing us to join it before it would otherwise be done.
                self.shutting_down = True
            self.wait_monitor_thread.join()
            self.wait_monitor_thread = None
            self.shutting_down = False
            if not abort and self.wait_monitor_thread_exception is not None:
                # Raise any unexpected errors from the wait monitor thread:
                _reraise(*self.wait_monitor_thread_exception)
            self.wait_monitor_thread_exception = None
            if not abort:
                # Don't want errors about incomplete task to be raised if we are aborting:
                self.CI_task.StopTask()
            if self.DO_task is not None:
                self.DO_task.StopTask()
        if self.CI_task is not None:
            self.CI_task.ClearTask()
            self.CI_task = None
        if self.DO_task is not None:
            self.DO_task.ClearTask()
            self.DO_task = None
        self.logger.debug('finished stop_tasks')

    def start_tasks(self):

        # The counter acquisition task:
        self.CI_task = Task()
        CI_chan = self.MAX_name + '/' + self.wait_acq_connection
        # What is the longest time in between waits, plus the timeout of the
        # second wait?
        interwait_times = np.diff([0] + list(self.wait_table['time']))
        max_measure_time = max(interwait_times + self.wait_table['timeout'])
        # Allow for software delays in timeouts.
        max_measure_time += 1.0
        min_measure_time = self.min_semiperiod_measurement
        self.logger.debug(
            "CI measurement range is: min: %f max: %f",
            min_measure_time,
            max_measure_time,
        )
        self.CI_task.CreateCISemiPeriodChan(
            CI_chan, '', min_measure_time, max_measure_time, DAQmx_Val_Seconds, ""
        )
        num_edges = 2 * (len(self.wait_table) + 1)
        self.CI_task.CfgImplicitTiming(DAQmx_Val_ContSamps, num_edges)
        self.CI_task.StartTask()

        # The timeout task:
        if self.wait_timeout_MAX_name is not None:
            self.DO_task = Task()
            DO_chan = self.wait_timeout_MAX_name + '/' + self.wait_timeout_connection
            self.DO_task.CreateDOChan(DO_chan, "", DAQmx_Val_ChanForAllLines)
            # Ensure timeout trigger is armed:
            written = int32()
            # Writing autostarts the task:
            self.DO_task.WriteDigitalLines(
                1, True, 1, DAQmx_Val_GroupByChannel, self.timeout_rearm, written, None
            )

    def compile_sections(self, h5file):
        
        # Get data from HDF5 file
        full_wait_table = []

        with h5py.File(h5file, "r") as hdf5_file:
            dataset = hdf5_file['waits']
            if len(dataset) == 0:
                # There are no waits. Do nothing.
                full_wait_table = None
            else:
                full_wait_table = dataset[:]

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
        for i in range(len(timestamps)-1):
            if full_wait_table is None:
                self.sections.append(None)
            else:
                section_wait_table = []
                for w in full_wait_table:
                    if timestamps[i] <= w['time'] < timestamps[i+1]:
                        w['time'] -= timestamps[i] # Adjust time for the subsection
                        section_wait_table.append(w)

                if len(section_wait_table) > 0:
                    self.sections.append(np.array(section_wait_table, dtype=full_wait_table.dtype))
                else:
                    self.sections.append(None)

    def run_section(self, section):

        start_t = time.perf_counter()
        self.wait_table = self.sections[section]
        self.semiperiods = []
        if self.wait_table is None:
            self.section_processed_waits.append([])
            return False
        self.start_tasks()

        # An array to store the results of counter acquisition:
        self.semiperiods = []
        self.wait_monitor_thread = threading.Thread(target=self.wait_monitor)
        # Not a daemon thread, as it implements wait timeouts - we need it to stay alive
        # if other things die.
        self.wait_monitor_thread.start()

        program_t = time.perf_counter() - start_t
        self.programming_times.append(program_t)

        return True


    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        self.logger.debug('transition_to_buffered')
        self.h5_file = h5file
        self.compile_sections(h5file)
        self.programming_times = []

        self.section_processed_waits = []

        self.run_section(0)
        self.runner.send_buffered()

        return {}

    def calc_wait_times(self, abort=False):

        if not abort and self.wait_table is not None:
            # Let's work out how long the waits were. The absolute times of each edge on
            # the wait monitor were:
            edge_times = np.cumsum(self.semiperiods)
            # Now there was also a rising edge at t=0 that we didn't measure:
            edge_times = np.insert(edge_times, 0, 0)
            # Ok, and the even-indexed ones of these were rising edges.
            rising_edge_times = edge_times[::2]
            # Now what were the times between rising edges?
            periods = np.diff(rising_edge_times)
            # How does this compare to how long we expected there to be between the
            # start of the experiment and the first wait, and then between each pair of
            # waits? The difference will give us the waits' durations.
            resume_times = self.wait_table['time']
            # Again, include the start of the experiment, t=0:
            resume_times = np.insert(resume_times, 0, 0)
            run_periods = np.diff(resume_times)
            wait_durations = periods - run_periods
            waits_timed_out = wait_durations > self.wait_table['timeout']

            # Work out how long the waits were, save them, post an event saying so:
            dtypes = [
                ('label', 'a256'),
                ('time', float),
                ('timeout', float),
                ('duration', float),
                ('timed_out', bool),
            ]
            data = np.empty(len(self.wait_table), dtype=dtypes)
            data['label'] = self.wait_table['label']
            data['time'] = self.wait_table['time']
            data['timeout'] = self.wait_table['timeout']
            data['duration'] = wait_durations
            data['timed_out'] = waits_timed_out

            return data

    def transition_to_manual(self, abort=False):
        self.logger.debug('transition_to_manual')
        self.stop_tasks(abort)

        if not abort and self.section_processed_waits is not None:

            processed_waits = 0
            for pws in self.section_processed_waits:
                processed_waits += len(pws)

            # Work out how long the waits were, save them, post an event saying so:
            dtypes = [
                ('section', int),
                ('label', 'a256'),
                ('time', float),
                ('timeout', float),
                ('duration', float),
                ('timed_out', bool),
            ]
            data = np.empty(processed_waits, dtype=dtypes)

            j = 0
            for i, proc_wait_section in enumerate(self.section_processed_waits):
                for w in proc_wait_section:
                    data[j]['section'] = i
                    data[j]['label'] = w['label']
                    data[j]['time'] = w['time']
                    data[j]['timeout'] = w['timeout']
                    data[j]['duration'] = w['duration']
                    data[j]['timed_out'] = w['timed_out']
                    j += 1

            with h5py.File(self.h5_file, 'a') as hdf5_file:
                hdf5_file.create_dataset('/data/waits', data=data)
                hdf5_file.create_dataset(f'/data/programming_time_wait_{self.device_name}', data=self.programming_times)
            self.wait_durations_analysed.post(self.h5_file)

        self.h5_file = None
        self.semiperiods = None
        return True

    def abort_buffered(self):
        self.runner.abort()
        return self.transition_to_manual(True)

    def abort_transition_to_buffered(self):
        self.runner.abort()
        return self.transition_to_manual(True)

    def program_manual(self, values):
        return {}
