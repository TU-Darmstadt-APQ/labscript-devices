
# --- Intermediate Device Imports 
# from labscript_devices import runviewer_parser, BLACS_tab
from labscript_utils.shared_drive import path_to_local
from labscript import config, IntermediateDevice, StaticAnalogQuantity, LabscriptError, set_passed_properties,StaticDDS
import numpy as np
import h5py

# --- Others
from .basic_device import RFGeneratorSpecs
from .agilent_4422B_device import agilent_e4422b_specs
from ..GPIBDevice import GPIBWorker, EosStrategy



# ---- STATIC DDS WRAPPER 
class AgilentE4422BRFOutput(StaticDDS):

    ''' These are methodes you are allowed to use in the experiment scipt'''

    description = "Agilent E4422B RF Output"

    def __init__(self, *args, **kwargs):
        StaticDDS.__init__(self, *args, **kwargs)
        self.rf_output = False

    def setfreq_khz(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("frequency must be int or float")

        freq_hz = float(value) * 1e3
        self.parent_device.specs.validate_frequency(freq_hz)
        self.setfreq(freq_hz)

    def setfreq_mhz(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("frequency must be int or float")

        freq_hz = float(value) * 1e6
        self.parent_device.specs.validate_frequency(freq_hz)
        self.setfreq(freq_hz)

    def setamp_dbm(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("power must be int or float")
        power_dbm = float(value)
        self.parent_device.specs.validate_power(power_dbm)
        self.setamp(power_dbm)

    def set_output_rf(self, state: bool):
        if not isinstance(state, bool):
            raise TypeError("state must be bool")
        self.rf_output = state


##############################################################################################################
#                                    Intermediate - DEVICE                                                   #
##############################################################################################################
class AgilentE4422B(IntermediateDevice):
    ''' Labscript device for one Agilent E4422B RF output. 
        This device controls one RF output (250e3 - 4e9 Hz). 
    
        Controllable static quantities:
            - frequency
            - power 
            - RF output ON/OFF 

        Example:
            rfgen = AgilentE4422B("rfgen", GPIB_address="GPIB0::5")
            rf = StaticDDS("rf", parent_device=rfgen, connection="rf")

            rf.setfreq(100e6)
            rf.setamp(-20)
            rfgen.set_output_rf(True)
        Args
            - name (str): Name of the device.
            - GPIB_address (str): GPIB address of the instrument. 
                                - Format: 
                                        1. "GPIB<number>::GPIB_address" 
                                        2. "ADAP::adapter_ip_address::GPIB_address" 
                                - Example:
                                        1. 'GPIB0::5' -> GPIB connection
                                        2. 'ADAP::192.168.123.132::5' -> Adapter connection
            - num_outputs (int): Number of output channels the instrument supports.

        NEW : Depending on the format of the passed GPIB_address : the device will try to connect through:
        -  a socket assuming a KOFOTRONIC adapter (Prologix alike) if GPIB_address = "ADAP::adapter_ip_address::GPIB_address"
        -  pyvisa assuming a normal GPIB connection if fromat GPIB_address =  "GPIB<number>::GPIB_address" 

    '''

    allowed_children = [AgilentE4422BRFOutput]
    description = 'AgilentE4422B'

    @set_passed_properties(property_names={"connection_table_properties": ["eos_strategy"]})
    def __init__(self, name, GPIB_address , eos_strategy = EosStrategy.LF.value , **kwargs):
        IntermediateDevice.__init__(self, name, None, **kwargs)
        self.BLACS_connection               = GPIB_address
        self.specs : RFGeneratorSpecs       = agilent_e4422b_specs 
        self.eos_strategy = eos_strategy


    def _get_rf_output_child(self):
        if len(self.child_devices) != 1:
            raise LabscriptError(
                f"{self.name} needs exactly one RF output child."
            )

        rf_output = self.child_devices[0]

        if not isinstance(rf_output, AgilentE4422BRFOutput):
            raise LabscriptError(
                f"{self.name} child must be AgilentE4422BRFOutput."
            )

        return rf_output

    def _get_output_state(self, rf_output):
        """
        Convert labscript child state into normalized device state.

        Important:
        StaticDDS frequency may default to 0.0.
        For this RF generator, 0 Hz is invalid, so we treat 0.0 as
        "frequency unchanged", not as a real programmed value.
        """

        raw_freq = rf_output.frequency.static_value
        raw_amp = rf_output.amplitude.static_value
        raw_rf_output = rf_output.rf_output

        frequency_hz = None
        power_dbm = None
        rf_output_state = None

        if raw_freq is not None:
            raw_freq = float(raw_freq)

            if raw_freq != 0.0:
                frequency_hz = raw_freq

        if raw_amp is not None:
            power_dbm = float(raw_amp)

        if raw_rf_output is not None:
            rf_output_state = bool(raw_rf_output)

        return {
            "frequency_hz": frequency_hz,
            "power_dbm": power_dbm,
            "rf_output": rf_output_state,
        }

    def _state_is_empty(self, state):
        return (
            state["frequency_hz"] is None
            and state["power_dbm"] is None
            and state["rf_output"] is None
        )

    def _validate_output_state(self, state):
        if state["frequency_hz"] is not None:
            self.specs.validate_frequency(state["frequency_hz"])

        if state["power_dbm"] is not None:
            self.specs.validate_power(state["power_dbm"])

    def _make_output_table(self, state):
        dtypes = [
            ("frequency_hz", np.float64),
            ("power_dbm", np.float64),
            ("rf_output", np.int8),
        ]

        output_table = np.zeros(1, dtype=dtypes)

        output_table["frequency_hz"] = (
            np.nan
            if state["frequency_hz"] is None
            else state["frequency_hz"]
        )

        output_table["power_dbm"] = (
            np.nan
            if state["power_dbm"] is None
            else state["power_dbm"]
        )

        # Tri-state RF output:
        # -1 = unchanged
        #  0 = OFF
        #  1 = ON
        output_table["rf_output"] = (
            -1
            if state["rf_output"] is None
            else int(state["rf_output"])
        )

        return output_table

    def generate_code(self, hdf5_file):
        IntermediateDevice.generate_code(self, hdf5_file)

        rf_output = self._get_rf_output_child()

        state = self._get_output_state(rf_output)

        if self._state_is_empty(state):
            return

        self._validate_output_state(state)

        output_table = self._make_output_table(state)

        grp = self.init_device_group(hdf5_file)

        grp.create_dataset(
            "OUTPUT_DATA",
            compression=config.compression,
            data=output_table,
        )