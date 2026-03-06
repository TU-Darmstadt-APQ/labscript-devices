from tqdm import tqdm
import pickle


class List_FPGA_instructions_helper:
    """
    This class is used to clculate instructions for the LIST-FPGA-Object.
    Therefore it is constructed as a wrapper. In order to be more easily be debuzgged this object takes a
    list-fpga-object as an input, that has to be created elsewhere (e.g. in ct-build or the main experimental script).
    The main advantage of using the List_FPGA_instructions_helper-class to create multiple instructions and trigger
    them via a single trigger pulse. Besides that through the hzelp of this object the programming in all is simplified:
        - seperate Freq.-changes, IO-changes
        - seperate instructions from the triggering mechanics (via software or hardware)
    Ideology behind this library steps to follow for implementing :
        1. think of the sequence you want to perform without caring for the trigger logic
        2. then decide when you actually nead a trigger pulse
    --> therefore trigger multiple instructions via a singular trigger pulse


    How the code works:
    Everytime a Freq. changes or a IO-pin is changed an entry in the objects `self.list_of_things`-list is made.
    An entry consists of the following information tuple:
        (
            [0] : absolute time
            [1] : type
            [2] : name
            [3] : position
            [4] : value
            [5] : delay (in FPGA-Clock-Cycles) <--- currently not used anymore!
        )
    Triggers are also recored there.
    Ech type of enty is represented by a type value (
        `type_DDS = 0`, `type_IO = 1`, `type_trigger_hardware = 2`, `type_wait_software = 3`
        )
    along with the absolute time this change schould ococour.
    The absolutre time might get computed indirectly: e.g. when the used issuing methode should perform the action asap=as soon as possible,
    these metrhodes as indicated by a methode name with an additional `_asap`-string attached. If a asap-methode is used,
    the absolute time of the previous instruction is used.
    Along with the time also a `name`-string-field is required: please filkl this field with an appropriate string name, this makes debugging
    of the system more easy afterwards.
    Besides the time, type and name entry the change might also require additional information like:
        - positional information (`position`): e.g. which DDS is changes or which IO-pin to change
        - a value (`value`): e.g. what is the new state of the DDS or the IO pin selected
        - the delay-entry is only here for historical reasons, this entry should not be used anymore might be dropped if future updates
    Any field that is not required should be 'blanked', by the `dummy_`-...-values provided in the class definition.


    After the filling of the list: self.list_of_things is completed this list is converted into LIST-FPGA-instructions, which are
    written to the LIST-FPGA-object in the methode self.convert2listFPGA()
    """

    # Here I'm defining some constants, please do NOT change them during runtime:

    dummy_value = -1  # to leave a field empty
    dummy_position = -1
    time_asap = -1

    type_DDS = 0
    type_IO = 1
    type_trigger_hardware = 2
    type_wait_software = 3

    """
    structure of self.list_of_things:
    (
        [0] absolute time
        [1] type
        [2] name
        [3] position
        [4] value
        [5] delay (in FPGA-Clock-Cycles)
    )
    self.list_of_things is defined in __init__()
    """

    _update_interval = 50

    def __init__(self):
        self.list_of_things = []
        self.list_of_things_time_only = []

    def _add_basic(self, absolute_time, type_it, name, position, value, delay):
        """
        a private methode for the appending procedure called by the publich methode for inserting list entries
        """
        self.list_of_things.append((absolute_time, type_it, name, position, value, delay))
        pass

    def add_freq(self,
                 absolute_time,
                 name,
                 dds_no,
                 freq_val_mhz,
                 delay=0):
        """
        This methode adds a frequency entry at a given time
        """
        delay_t_this_event = self._compute_delays_backwards(delay)
        expected_true_time_this_event = absolute_time + delay_t_this_event

        self._add_basic(expected_true_time_this_event,
                        self.type_DDS,
                        name,
                        int(dds_no),
                        freq_val_mhz,
                        0
                        )
        pass

    def add_freq_asap(self,
                      name,
                      dds_no,
                      freq_val_mhz,
                      delay=0):
        """
        This methode adds a frequency entry as soon as possbile in the time: this
        entry will be converted to perform an instruction that is executed directly
        after the first instruction is executed.
        This methode cannot be called to create the first instruction: because there
        would be no prior instruction to derive the time from
        """
        expected_time_previous = self._get_time_of_last_event_issued()
        delay_t_this_event = self._compute_delays_backwards(delay)
        expected_true_time_this_event = expected_time_previous + delay_t_this_event

        self._add_basic(expected_true_time_this_event,
                        self.type_DDS,
                        name,
                        int(dds_no),
                        freq_val_mhz,
                        0
                        )
        pass

    def _get_time_of_last_event_issued(self):
        """This methode returns the timing set for the last event that has been issued.
        Therefore this methode looks up the last entry in the self.list_of_things of this object
        and fetches the last entry. From this last entry the indey [0] is stripped and returned in order
        return the absolute timing the last event has been issued (without taking delays into account).
        In the case this operation fails due to problem, that the self.list_of_things is an empty list an exception
        will be raised.
        """
        if len(self.list_of_things) > 0:
            last_entry = self.list_of_things[-1]
            return last_entry[0]
            pass
        else:
            # no prevoius element availabel
            raise Exception("Could not fetch the timing of a prevoius element, because the list self.list_of_things is empty.\
                            This Error is likely rised due to the fact, that an object from the class List_FPGA_instructions_helper\
                            is used with some sort of '...\a\sap' before any absolute function is called like .add_freq() or .switch() or \
                            .trigger_hardware(). In order to solve this issue, call one of these methodes in front of or instead of \
                            the called '...\a\sap'-mathode")
        pass

    def switch(self, absolute_time, name, io_pin, pin_val, delay=0):
        """
        Used to switch a SINGLE IO-pin to a specific value at a given time.
        """
        delay_t_this_event = self._compute_delays_backwards(delay)
        expected_true_time_this_event = absolute_time + delay_t_this_event

        self._add_basic(expected_true_time_this_event,
                        self.type_IO,
                        name,
                        int(io_pin),
                        bool(pin_val),
                        0
                        )
        pass

    def switch_asap(self, name, io_pin, pin_val, delay=0):
        """Used to switch a SINGLE IO-pin to a specific valueas soon as possbile in the time: this
        entry will be converted to perform an instruction that is executed directly
        after the first instruction is executed.
        This methode cannot be called to create the first instruction: because there
        would be no prior instruction to derive the time from
        """
        expected_time_previous = self._get_time_of_last_event_issued()
        delay_t_this_event = self._compute_delays_backwards(delay)
        expected_true_time_this_event = expected_time_previous + delay_t_this_event

        self._add_basic(expected_true_time_this_event,
                        self.type_IO,
                        name,
                        int(io_pin),
                        bool(pin_val),
                        0
                        )
        pass

    def wait_software(self, name):
        """let the FPGA wait for a software trigger event"""
        expected_time_previous = self._get_time_of_last_event_issued()

        self._add_basic(expected_time_previous,
                        self.type_wait_software,
                        name,
                        self.dummy_position,
                        self.dummy_value,
                        0
                        )
        pass

    def trigger_hardware(self, absolute_time, name):
        """trigger the FPGA via a harware trigger at a given time: when this event is later
        converted into instruction the pulse blaster is automatically probrammed to provide a
        pulse at the required time."""
        self._add_basic(absolute_time,
                        self.type_trigger_hardware,
                        name,
                        self.dummy_position,
                        self.dummy_value,
                        0
                        )
        pass

    def end_FPGA(self, name):
        """This creates a software trigger, used at the end of the FPGA-usage, but
        the actual software trigger is never issued to the FPGA: this sets the FPGA into a wait
        position after, so the FPGA can the be programmed prior to the next experimental shot
        """
        expected_time_previous = self._get_time_of_last_event_issued()

        self._add_basic(expected_time_previous,
                        self.type_wait_software,
                        name,
                        self.dummy_position,
                        self.dummy_value,
                        0
                        )
        pass

    def _compute_delays_rounded_from_delayt(self, delay_t):
        """
        Computes delays from seconds into delays of the FPGA-internal-clock cycles.
        """
        if delay_t > 60e-9:
            round(delay_t, 5)
            return int(round((delay_t - 60e-9) / 20e-9))
        else:
            return 0

    def _compute_delays_backwards(self, delay):
        """Computes a given delay in FPGA-cycles back into a delay that is in the 'seconds'-domain"""
        if delay > 0:
            return (delay * 20e-9) + 60e-9 # add round (,5) here??
        else:
            return 0

    def _write_debug2pickleRAW(self):
        """Used internally to write a debugging file to the disk of the self.list_of_things-attribute
        The created pickle-file only uses python build in types and therefore this file
        can be opened into any python environment.
        """
        _PATH = "Z:\\QUIPS-B\\debug_FPGA\\list_of_things.pckl"
        with open(_PATH, "wb") as pckl_file:
            pickle.dump(self.list_of_things, pckl_file)
        print(f"wrote self.list_of_things to {_PATH}")

    def _write_debug2pickleSorted(self, sorted_list):
        """Used internally to write a debugging file to the disk of
        the self.list_of_things-attribute but after an essential sorting step
        The created pickle-file only uses python build in types and therefore this file
        can be opened into any python environment.
        """
        _PATH = "Z:\\QUIPS-B\\debug_FPGA\\list_of_things_sorted.pckl"
        with open(_PATH, "wb") as pckl_file:
            pickle.dump(sorted_list, pckl_file)
        print(f"wrote self.list_of_things to {_PATH}")

    def _write_debug2pickleFPGA(self, list_FPGA_obj):
        """Used internally to write a debugging file to the disk of
        the the finally created instructions.
        The created pickle-file only uses python build in types and therefore this file
        can be opened into any python environment.
        """
        _PATH = "Z:\\QUIPS-B\\debug_FPGA\\instructions.pckl"
        with open(_PATH, "wb") as pckl_file:
            pickle.dump(list_FPGA_obj.instructions, pckl_file)
        print(f"wrote self.list_of_things to {_PATH}")

    def convert2listFPGA(self,
                         list_FPGA_obj,
                         warn_delay_max_s=10e-3):
        """converts the object in order to send instructions to the provided list_FPGA_obj"""

        previous_hardware_trigger_called = False
        previous_hardware_trigger_time = None
        # print(f"self.list_of_things=\n{self.list_of_things}")

        print(f"computing List_FPGA_instructions_helper.convert2listFPGA() to do: {len(self.list_of_things)} number entries")

        self._write_debug2pickleRAW()

        # sort the list by timing:
        sorted_timings_list = sorted(self.list_of_things, key=lambda entry: entry[0])  # the lambdafunction enables the sorting by the time = entry[0]

        self._write_debug2pickleSorted(sorted_timings_list)

        for i, entry in enumerate(sorted_timings_list):
            # go through the self.list_of_things elementwise & create the instructions:
            # print(f"i={i} entry={entry}")
            # first element should contain an trigger:
            if i == 0 and entry[1] != self.type_trigger_hardware:
                print(i, entry)
                raise Exception(f"The first event for the List_FPGA_instructions_helper should be a self.trigger_hardware(). To fix this issue go above {entry[2]} and insert an .trigger_hardware()")

            # take care of the timing
            if entry[0] == -1:
                # self.list_of_things_time_only.append(self.list_of_things_time_only[-1])
                expected_time = self.list_of_things_time_only[-1]
                pass
            else:
                # self.list_of_things_time_only.append(entry[0])
                expected_time = entry[0]
                pass

            # take care of the different device classes
            if entry[1] == self.type_DDS:
                frequency = entry[4]
                DDS_Nr = entry[3]
                if previous_hardware_trigger_called == True:
                    trigger_type = list_FPGA_obj.LIST_FPGA_TRIGGER_WAIT_HARDWARE
                    delay_t = expected_time - self.list_of_things_time_only[-1]
                else:
                    trigger_type = list_FPGA_obj.LIST_FPGA_TRIGGER_NO_WAIT
                    delay_t = expected_time - self.list_of_things_time_only[-1]
                    if warn_delay_max_s < (expected_time - previous_hardware_trigger_time):
                        raise Exception(f"the FPGA has been isued by an delay_t={delay_t}s more than the maximum allowed warn_delay_max_s={warn_delay_max_s}s please put between 'instruction' {self.list_of_things[i-1][2]} and {entry[2]} a self.trigger_hardware()")

                # compute delay in FPGA-cycles
                delay = round(entry[5] + self._compute_delays_rounded_from_delayt(delay_t))

                # now add an instruction
                list_FPGA_obj.add_only_frequency(
                    frequency,
                    DDS_Nr,
                    delay,
                    trigger_type)
                previous_hardware_trigger_called = False
                pass
            elif entry[1] == self.type_IO:
                frequency = list_FPGA_obj.DDS_NONE_FREQ
                DDS_Nr = list_FPGA_obj.DDS_NONE

                if previous_hardware_trigger_called == True:
                    trigger_type = list_FPGA_obj.LIST_FPGA_TRIGGER_WAIT_HARDWARE
                    delay_t = expected_time - self.list_of_things_time_only[-1]
                else:
                    trigger_type = list_FPGA_obj.LIST_FPGA_TRIGGER_NO_WAIT
                    delay_t = expected_time - self.list_of_things_time_only[-1]
                    if warn_delay_max_s < (expected_time - previous_hardware_trigger_time):
                        raise Exception(f"the FPGA has been isued by an delay_t={delay_t}s more than the maximum allowed warn_delay_max_s={warn_delay_max_s}s please put between 'instruction' {self.list_of_things[i-1][2]} and {entry[2]} a self.trigger_hardware()")

                delay = entry[5] + self._compute_delays_rounded_from_delayt(delay_t)
                if delay < 0:
                    raise Exception(f"Issued Delay that is negative, entry: {entry}, index={i}")
                digital_out_id = entry[3]
                digital_out_val = entry[4]

                # now add an instruction
                list_FPGA_obj.add_switch(delay,
                                         trigger_type,
                                         digital_out_id,
                                         digital_out_val)
                previous_hardware_trigger_called = False
                pass
            elif entry[1] == self.type_trigger_hardware:
                previous_hardware_trigger_called = True
                previous_hardware_trigger_time = entry[0]
                list_FPGA_obj.trigger(expected_time)
                delay = 0
                pass
            elif entry[1] == self.type_wait_software:
                list_FPGA_obj.software_wait_trigger()
                delay = 0
                pass
            else:
                raise Exception("unkown type provided, check the list of supportet types @ the top (class definition of the 'List_FPGA_instructions_helper'-class)")
                pass

            self.list_of_things_time_only.append(expected_time + self._compute_delays_backwards(entry[5]))
            if entry[5] != 0:
                raise Exception(f"entry = {entry} has delay!")
        self._write_debug2pickleFPGA(list_FPGA_obj)
        pass
