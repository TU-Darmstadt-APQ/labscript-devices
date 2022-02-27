#####################################################################
#                                                                   #
# /labscript_devices/PrawnBlaster/blacs_worker.py                   #
#                                                                   #
# Copyright 2021, Philip Starkey                                    #
#                                                                   #
# This file is part of labscript_devices, in the labscript suite    #
# (see http://labscriptsuite.org), and is licensed under the        #
# Simplified BSD License. See the license.txt file in the root of   #
# the project for the full license.                                 #
#                                                                   #
#####################################################################
import time
import labscript_utils.h5_lock
import h5py
from blacs.tab_base_classes import Worker
from labscript_utils.connections import _ensure_str
import labscript_utils.properties as properties
import zmq
import threading
import numpy as np


class PrawnBlasterWorker(Worker):
    """The primary worker for the PrawnBlaster.

    This worker handles configuration and communication
    with the hardware.
    """

    def init(self):
        """Initialises the hardware communication.

        This function is automatically called by BLACS
        and configures hardware communication with the device.
        """

        # fmt: off
        global h5py; import labscript_utils.h5_lock, h5py
        global serial; import serial
        global time; import time
        global re; import re
        global numpy; import numpy
        global zprocess; import zprocess
        self.smart_cache = {}
        self.cached_pll_params = {}
        self.run_thread = None
        # fmt: on

        self.all_waits_finished = zprocess.Event("all_waits_finished", type="post")
        self.wait_durations_analysed = zprocess.Event(
            "wait_durations_analysed", type="post"
        )
        self.wait_completed = zprocess.Event("wait_completed", type="post")
        self.current_wait = 0
        self.wait_table = None
        self.measured_waits = None
        self.wait_timeout = None
        self.h5_file = None
        self.started = False

        self.start_event = threading.Event()

        self.prawnblaster = serial.Serial(self.com_port, 115200, timeout=1)
        self.serial_lock = threading.Lock()
        self.check_status()

        # configure number of pseudoclocks
        with self.serial_lock:
            self.prawnblaster.write(b"setnumpseudoclocks %d\r\n" % self.num_pseudoclocks)
            assert self.prawnblaster.readline().decode() == "ok\r\n"

        # Configure pins
        with self.serial_lock:
            for i, (out_pin, in_pin) in enumerate(zip(self.out_pins, self.in_pins)):
                self.prawnblaster.write(b"setoutpin %d %d\r\n" % (i, out_pin))
                assert self.prawnblaster.readline().decode() == "ok\r\n"
                self.prawnblaster.write(b"setinpin %d %d\r\n" % (i, in_pin))
                assert self.prawnblaster.readline().decode() == "ok\r\n"
        
        self.context = zmq.Context()
        self.from_master_socket = self.context.socket(zmq.SUB)
        self.to_master_socket = self.context.socket(zmq.PUB)

        self.from_master_socket.connect(f"tcp://localhost:44555")
        self.to_master_socket.connect(f"tcp://localhost:44556")

        self.from_master_socket.subscribe("")

        self.sections = []

    def check_status(self):
        """Checks the operational status of the PrawnBlaster.

        This is automatically called by BLACS to update the status
        of the PrawnBlaster. It also reads the lengths of any 
        accumulated waits during a shot.

        Returns:
            (int, int, bool): Tuple containing:

            - **run_status** (int): Possible values are: 

                * 0 : manual-mode
                * 1 : transitioning to buffered execution
                * 2 : buffered execution
                * 3 : abort requested
                * 4 : currently aborting buffered execution
                * 5 : last buffered execution aborted
                * 6 : transitioning to manual mode

            - **clock_status** (int): Possible values are:

                * 0 : internal clock
                * 1 : external clock

            - **waits_pending** (bool): Indicates if all expected waits have
              not been read out yet.
        """

        if (
            self.started
            and self.wait_table is not None
            and self.current_wait < len(self.wait_table)
        ):
            # Try to read out wait. For now, we're only reading out waits from
            # pseudoclock 0 since they should all be the same (requirement imposed by labscript)
            
            with self.serial_lock:
                self.prawnblaster.write(b"getwait %d %d\r\n" % (0, self.current_wait))
                response = self.prawnblaster.readline().decode()
            if response != "wait not yet available\r\n":
                # Parse the response from the PrawnBlaster
                wait_remaining = int(response)
                # Divide by two since the clock_resolution is for clock pulses, which
                # have twice the clock_resolution of waits
                # Technically, waits also only have a resolution of `clock_resolution`
                # but the PrawnBlaster firmware accepts them in half of that so that
                # they are easily converted to seconds via the clock frequency.
                # Maybe this was a mistake, but it's done now.
                clock_resolution = self.device_properties["clock_resolution"] / 2
                input_response_time = self.device_properties["input_response_time"]
                timeout_length = round(
                    self.wait_table[self.current_wait]["timeout"] / clock_resolution
                )

                if wait_remaining == (2 ** 32 - 1):
                    # The wait hit the timeout - save the timeout duration as wait length
                    # and flag that this wait timedout
                    self.measured_waits[self.current_wait] = (
                        timeout_length * clock_resolution
                    )
                    self.wait_timeout[self.current_wait] = True
                else:
                    # Calculate wait length
                    # This is a measurement of between the end of the last pulse and the
                    # retrigger signal. We obtain this by subtracting off the time it takes
                    # to detect the pulse in the ASM code once the trigger has hit the input
                    # pin (stored in input_response_time)
                    self.measured_waits[self.current_wait] = (
                        (timeout_length - wait_remaining) * clock_resolution
                    ) - input_response_time
                    self.wait_timeout[self.current_wait] = False

                self.logger.info(
                    f"Wait {self.current_wait} finished. Length={self.measured_waits[self.current_wait]:.9f}s. Timed-out={self.wait_timeout[self.current_wait]}"
                )

                # Inform any interested parties that a wait has completed:
                self.wait_completed.post(
                    self.h5_file,
                    data=_ensure_str(self.wait_table[self.current_wait]["label"]),
                )

                # increment the wait we are looking for!
                self.current_wait += 1

                # post message if all waits are done
                if len(self.wait_table) == self.current_wait:
                    self.logger.info("All waits finished")
                    self.all_waits_finished.post(self.h5_file)

        # Determine if we are still waiting for wait information
        waits_pending = False
        if self.wait_table is not None:
            if self.current_wait == len(self.wait_table):
                waits_pending = False
            else:
                waits_pending = True

        run_status, clock_status = self.read_status()
        return run_status, clock_status, waits_pending

    def read_status(self, Total = False):
        """Reads the status of the PrawnBlaster.

        Returns:
            (int, int): Tuple containing

                - **run-status** (int): Run status code
                - **clock-status** (int): Clock status code
        """

        if (not Total) and (self.run_thread is not None):
            if self.run_thread.is_alive():
                print("run_thread is alive. Dont check status")
                return 2, 0 # TODO

        with self.serial_lock:
            self.prawnblaster.write(b"status\r\n")
            response = self.prawnblaster.readline().decode()
        match = re.match(r"run-status:(\d) clock-status:(\d)(\r\n)?", response)
        if match:
            return int(match.group(1)), int(match.group(2))
        elif response:
            raise Exception(
                f"PrawnBlaster is confused: saying '{response}' instead of 'run-status:<int> clock-status:<int>'"
            )
        else:
            raise Exception(
                f"PrawnBlaster is returning a invalid status '{response}'. Maybe it needs a reboot."
            )

    def program_manual(self, values):
        """Manually sets the state of output pins for the pseudoclocks.

        Args:
            values (dict): Dictionary of pseudoclock: value pairs to set.

        Returns:
            dict: `values` from arguments on successful programming
            reflecting current output state.
        """

        for channel, value in values.items():
            with self.serial_lock:
                pin = int(channel.split()[1])
                pseudoclock = self.out_pins.index(pin)
                if value:
                    self.prawnblaster.write(b"go high %d\r\n" % pseudoclock)
                else:
                    self.prawnblaster.write(b"go low %d\r\n" % pseudoclock)

                assert self.prawnblaster.readline().decode() == "ok\r\n"

        return values

    def run_experiment(self, device_name):

        self.from_master_socket.recv()

        self.start_event.wait()
        while True:
            while True:
                time.sleep(0.001)
                run_status, clock_status = self.read_status(True)
                if run_status == 0:
                    break

            self.to_master_socket.send(str.encode(f"fin {device_name}"))

            msg = self.from_master_socket.recv() # load message

            if msg == b'exit':
                return

            next_section = int(msg.split()[-1])
            print(f"Next section is {next_section}")

            self.program_clock(next_section)

            self.to_master_socket.send(str.encode(f"rdy {device_name}"))
            msg = self.from_master_socket.recv() # load message

            self.start_run()


    def program_clock(self, section):
        # Program instructions
        for pseudoclock, pulse_program in enumerate(self.sections[section]['pulse_program']):
            for i, instruction in enumerate(pulse_program):
                if i == len(self.smart_cache[pseudoclock]):
                    # Pad the smart cache out to be as long as the program:
                    self.smart_cache[pseudoclock].append(None)

                # Only program instructions that differ from what's in the smart cache:
                if self.smart_cache[pseudoclock][i] != instruction:
                    
                    with self.serial_lock:
                        self.prawnblaster.write(
                            b"set %d %d %d %d\r\n"
                            % (
                                pseudoclock,
                                i,
                                instruction["half_period"],
                                instruction["reps"],
                            )
                        )
                        response = self.prawnblaster.readline().decode()
                    assert (
                        response == "ok\r\n"
                    ), f"PrawnBlaster said '{response}', expected 'ok'"
                    self.smart_cache[pseudoclock][i] = instruction

        if not self.is_master_pseudoclock:
            # Start the Prawnblaster and have it wait for a hardware trigger
            self.wait_for_trigger()


    def filter_data_by_time(self, time, values, min_time, max_time):

        if time is None:
            return None
        if values is None:
            return None

        return_values = []
        for i in range(len(values)):
            if min_time <= time[i] <= max_time:
                return_values.append(values[i])
        return np.array(return_values)

    def compile_sections(self, h5file, device_name):
        
        # Get data from HDF5 file
        pulse_programs = []
        pulse_program_times = []
        with h5py.File(h5file, "r") as hdf5_file:
            group = hdf5_file[f"devices/{device_name}"]
            for i in range(self.num_pseudoclocks):
                pulse_programs.append(group[f"PULSE_PROGRAM_{i}"][:])
                pulse_program_times.append(group[f"TIMES_{i}"][:])
                self.smart_cache.setdefault(i, [])
            self.device_properties = labscript_utils.properties.get(
                hdf5_file, device_name, "device_properties"
            )
            self.is_master_pseudoclock = self.device_properties["is_master_pseudoclock"]

            # waits
            # dataset = hdf5_file["waits"]
            # acquisition_device = dataset.attrs["wait_monitor_acquisition_device"]
            # timeout_device = dataset.attrs["wait_monitor_timeout_device"]
            # if (
            #     len(dataset) > 0
            #     and acquisition_device
            #     == "%s_internal_wait_monitor_outputs" % device_name
            #     and timeout_device == "%s_internal_wait_monitor_outputs" % device_name
            # ):
            #     self.wait_table = dataset[:]
            #     self.measured_waits = numpy.zeros(len(self.wait_table))
            #     self.wait_timeout = numpy.zeros(len(self.wait_table), dtype=bool)
            # else:
            #     self.wait_table = (
            #         None  # This device doesn't need to worry about looking at waits
            #     )
            #     self.measured_waits = None
            #     self.wait_timeout = None

            jumps = hdf5_file['jumps'][:]
            end_time = hdf5_file['devices']['PB'].attrs['stop_time']
            
        timestamps = []
        for j in range(len(jumps)):
            timestamps.append(jumps[j]["time"])
            timestamps.append(jumps[j]["to_time"])

        timestamps.append(0)
        timestamps.append(end_time) # TODO: final time

        timestamps = sorted(set(timestamps))

        self.sections = []
        for i in range(len(timestamps)-1):

            print(f"Generate section {i}")

            start = timestamps[i]
            end = timestamps[i+1]
            
            # Just overwrite every time as we are only interested in the last ones...
            # DO_final_values, DO_task, DO_all_zero = self.program_buffered_DO(self.filter_data_by_time(times_table, DO_table, start, end))
            # AO_final_values, AO_task, AO_all_zero = self.program_buffered_AO(self.filter_data_by_time(times_table, AO_table, start, end))

            current_pulse_program = []

            for j in range(len(pulse_programs)):
                pp = pulse_programs[j]
                pt = pulse_program_times[j]
                current_pulse_program.append(self.filter_data_by_time(pt, pp, start, end))


            section = {
                "start": start,
                "end": end,
                "pulse_program": current_pulse_program
            }


            self.sections.append(section)


    def transition_to_buffered(self, device_name, h5file, initial_values, fresh, intercom):
        print("Transition to buffered")
        """Configures the PrawnBlaster for buffered execution.

        Args:
            device_name (str): labscript name of PrawnBlaster
            h5file (str): path to shot file to be run
            initial_values (dict): Dictionary of output states at start of shot
            fresh (bool): When `True`, clear the local :py:attr:`smart_cache`, forcing
                a complete reprogramming of the output table.

        Returns:
            dict: Dictionary of the expected final output states.
        """

        if fresh:
            self.smart_cache = {}

        # fmt: off
        self.h5_file = h5file  # store reference to h5 file for wait monitor
        self.current_wait = 0  # reset wait analysis
        self.started = False   # Prevent status check from detecting previous wait values
        self.start_event.clear()
        #                        betwen now and when we actually send the start signal
        # fmt: on

        self.compile_sections(h5file, device_name)

        # # Configure clock from device properties
        clock_mode = 0
        if self.device_properties["external_clock_pin"] is not None:
            if self.device_properties["external_clock_pin"] == 20:
                clock_mode = 1
            elif self.device_properties["external_clock_pin"] == 22:
                clock_mode = 2
            else:
                raise RuntimeError(
                    f"Invalid external clock pin {self.device_properties['external_clock_pin']}. Pin must be 20, 22 or None."
                )
        clock_frequency = self.device_properties["clock_frequency"]

        # Now set the clock details
        with self.serial_lock:
            self.prawnblaster.write(b"setclock %d %d\r\n" % (clock_mode, clock_frequency))
            response = self.prawnblaster.readline().decode()
        assert response == "ok\r\n", f"PrawnBlaster said '{response}', expected 'ok'"

        self.program_clock(0)

        self.run_thread = threading.Thread(target=self.run_experiment, args=(device_name,))
        self.run_thread.start()


        # All outputs end on 0
        final = {}
        for pin in self.out_pins:
            final[f"GPIO {pin:02d}"] = 0
        return final

    def start_run(self):
        """When used as the primary pseudoclock, starts execution
        in software time to engage the shot."""

        # Start in software:
        self.logger.info("sending start")
        with self.serial_lock:
            self.prawnblaster.write(b"start\r\n")
            response = self.prawnblaster.readline().decode()
        assert response == "ok\r\n", f"PrawnBlaster said '{response}', expected 'ok'"

        # set started = True
        self.started = True
        self.start_event.set()

    def wait_for_trigger(self):
        """When used as a secondary pseudoclock, sets the PrawnBlaster
        to wait for an initial hardware trigger to begin execution."""

        # Set to wait for trigger:
        self.logger.info("sending hwstart")
        with self.serial_lock:
            self.prawnblaster.write(b"hwstart\r\n")
            response = self.prawnblaster.readline().decode()
        assert response == "ok\r\n", f"PrawnBlaster said '{response}', expected 'ok'"

        running = False
        while not running:
            run_status, clock_status = self.read_status()
            # If we are running, great, the PrawnBlaster is waiting for a trigger
            if run_status == 2:
                running = True
            # if we are not in TRANSITION_TO_RUNNING, then something has gone wrong
            # and we should raise an exception
            elif run_status != 1:
                raise RuntimeError(
                    f"Prawnblaster did not return an expected status. Status was {run_status}"
                )
            time.sleep(0.01)

        # set started = True
        self.started = True

    def transition_to_manual(self):
        """Transition the PrawnBlaster back to manual mode from buffered execution at
        the end of a shot.

        Returns:
            bool: `True` if transition to manual is successful.
        """

        self.run_thread.join()

        if self.wait_table is not None:
            with h5py.File(self.h5_file, "a") as hdf5_file:
                # Work out how long the waits were, save em, post an event saying so
                dtypes = [
                    ("label", "a256"),
                    ("time", float),
                    ("timeout", float),
                    ("duration", float),
                    ("timed_out", bool),
                ]
                data = numpy.empty(len(self.wait_table), dtype=dtypes)
                data["label"] = self.wait_table["label"]
                data["time"] = self.wait_table["time"]
                data["timeout"] = self.wait_table["timeout"]
                data["duration"] = self.measured_waits
                data["timed_out"] = self.wait_timeout

                self.logger.info(str(data))

                hdf5_file.create_dataset("/data/waits", data=data)

            self.wait_durations_analysed.post(self.h5_file)

        # If PrawnBlaster is master pseudoclock, then it will have it's status checked
        # in the BLACS tab status check before any transition to manual is called.
        # However, if it's not the master pseudoclock, we need to check here instead!
        if not self.is_master_pseudoclock:
            # Wait until shot completes
            while True:
                run_status, clock_status = self.read_status()
                if run_status == 0:
                    break
                if run_status in [3, 4, 5]:
                    raise RuntimeError(
                        f"Prawnblaster status returned run-status={run_status} during transition to manual"
                    )
                time.sleep(0.01)

        return True

    def shutdown(self):
        """Cleanly shuts down the connection to the PrawnBlaster hardware."""

        with self.serial_lock:
            self.prawnblaster.close()

        self.from_master_socket.close()
        self.to_master_socket.close()

        self.context.term()

    def abort_buffered(self):
        """Aborts a currently running buffered execution.

        Returns:
            bool: `True` is abort is successful.
        """
        if not self.is_master_pseudoclock:
            # Only need to send abort signal if we have told the PrawnBlaster to wait
            # for a hardware trigger. Otherwise it's just been programmed with
            # instructions and there is nothing we need to do to abort.
            
            with self.serial_lock:
                self.prawnblaster.write(b"abort\r\n")
                assert self.prawnblaster.readline().decode() == "ok\r\n"
            # loop until abort complete
            while self.read_status()[0] != 5:
                time.sleep(0.5)
        return True

    def abort_transition_to_buffered(self):
        """Aborts a transition to buffered.

        Calls :py:meth:`abort_buffered`.
        """

        return self.abort_buffered()
