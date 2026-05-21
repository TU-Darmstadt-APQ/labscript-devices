from enum import Enum
from operator import add
from basic_device import RFGeneratorSpecs, NumericSpec, UnitFreq, UnitPower
import pyvisa as visa
from typing import List, Tuple


# ------------------------------------------------------ SPECS
agilent_e4422b_specs = RFGeneratorSpecs(
    manufact    = "Agilent",
    model       = "E4422B",
    series      = "ESG-A",

    frequency           = NumericSpec(250e3, 4e9, "Hz"),
    power_dbm           = NumericSpec(-136.0, 7.0, "dBm"),
    sweep_points        = NumericSpec(2, 401),
    sweep_dwell_s       = NumericSpec(0.001, 60.0, "s"),
    am_depth_percent    = NumericSpec(0.0, 100.0, "%"),
)


# ------------------------------------------------------ ENUMS 
class ModeShape(str, Enum):
    SINE       = "SINE"
    TRIANGLE   = "TRI"
    SQUARE     = "SQU"
    RAMP       = "RAMP"
    NOISE      = "NOIS"
    DUAL_SINE  = "DUAL"
    SWEPT_SINE = "SWEP"

class ModeCoupling(str, Enum):
    ''' Coupling Mode used by Modulation Paths'''
    AC = "AC"
    DC = "DC"

class ModeFreq(str, Enum):
    CW   = "CW"  # Continuous Wave
    LIST = "LIST"

class ModeSource(str, Enum):
    '''Source Mode used by Moduation Paths'''
    INT  = "INT"
    EXT1 = "EXT1"
    EXT2 = "EXT2"

class ModeSweep(str, Enum):
    """Sweep Mode used by the signal generator."""
    STEP = "STEP"  # Step sweep
    LIST = "LIST"  # List sweep

class ModeSlope(str, Enum):
    """Trigger slope / edge direction."""
    POS = "POS"  # Positive / rising edge
    NEG = "NEG"  # Negative / falling edge

class ModeTrig(str, Enum):
    """SCPI sweep trigger sources."""
    BUS = "BUS"  # remote bus trigger
    IMM = "IMM"  # immediate trigger
    EXT = "EXT"  # external TRIGGER IN
    KEY = "KEY"  # front-panel Trigger key

class ModePower(str, Enum):
    FIX  = "FIX"
    LIST = "LIST"


# ------------------------------------------------------ The Class
class AgilentE4422BDevice:
    """ Minimal SCPI wrapper for Agilent/Keysight E4422B ESG signal generator. """

    specs = agilent_e4422b_specs


    def __init__(self, write, query , addr = None):
        self.write = write
        self.query = query
        
        self.GPIB_address = addr

        if addr:
            self._device_init()


    def _device_init(self):
        try:
            # self.GPIB_address = self.GPIB_address +"::INSTR"
            self.rm = visa.ResourceManager()
            self.GPIB_connection = self.rm.open_resource(self.GPIB_address)
            self.GPIB_connection.timeout = 5000  # ms
            self.GPIB_connection.lock(requested_key=self.GPIB_address)   
            self.GPIB_connection.lock_excl()
            self.write = self.GPIB_connection.write
            self.query = self.GPIB_connection.query

            print(self.identify())

        except Exception as e:
            raise Exception(f"Error with pyvisa connection : {e}")
        


    # -------------------------------------------------- Helpers
    @staticmethod
    def _bool(state: bool) -> str:
        if not isinstance(state, bool):
            raise TypeError("State must be bool")
        return "ON" if state else "OFF"
    
    @staticmethod
    def _byte(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("value must be int")
        if not (0 <= value <= 255):
            raise ValueError("value must be between 0 and 255")
        return value

    @staticmethod
    def _path(path: int) -> int:
        """Validate path number."""
        if path not in (1, 2):
            raise ValueError("Path must be 1 or 2")
        return path
    
    def _query_bool(self, command: str) -> bool:
        return bool(int(float(self.query(command).strip())))

    def _query_float(self, command: str) -> float:
        return float(self.query(command).strip())
    
    # -------------------------------------------------- Common Commands
    def identify(self) -> str:
        return self.query("*IDN?").strip()

    def reset(self) -> None:
        ''' Writes *RST '''
        self.write("*RST")

    def self_test(self) -> bool:
        """ Return True if self-test passed. SCPI *TST? returns 0 when the test passes."""
        return int(self.query("*TST?").strip()) == 0

    def wait(self) -> None:
        self.write("*WAI")

    def operation_complete(self) -> bool:
        """Return True when all pending operations are complete."""
        return int(self.query("*OPC?").strip()) == 1
    
    # -------------------------------------------------- Register Commands
    def clear_status(self) -> None:
        """Clear status registers and error/event queues."""
        self.write("*CLS")

    def get_status_byte(self) -> int:
        """Read the status byte register."""
        return int(float(self.query("*STB?").strip()))

    def get_service_request_enable(self) -> int:
        """Read the service request enable register."""
        return int(float(self.query("*SRE?").strip()))

    def set_service_request_enable(self, value: int) -> None:
        """Set the service request enable register."""
        value = self._byte(value)
        self.write(f"*SRE {value}")

    def get_standard_event_status(self) -> int:
        """Read and clear the standard event status register."""
        return int(float(self.query("*ESR?").strip()))

    def get_standard_event_enable(self) -> int:
        """Read the standard event status enable register."""
        return int(float(self.query("*ESE?").strip()))

    def set_standard_event_enable(self, value: int) -> None:
        """Set the standard event status enable register."""
        value = self._byte(value)
        self.write(f"*ESE {value}")
        
    # -------------------------------------------------- Output
    def get_output_rf(self) -> bool:
        return self._query_bool(":OUTPut:STATe?")

    def set_output_rf(self, state: bool) -> None:
        """Turn RF output on or off."""
        state = self._bool(state)
        self.write(f":OUTPut:STATe {state}")

    def get_output_modulation(self) -> bool:
        return self._query_bool(":OUTPut:MODulation:STATe?")

    def set_output_modulation(self, state: bool) -> None:
        """Turn modulation output on or off."""
        state = self._bool(state)
        self.write(f":OUTPut:MODulation:STATe {state}")

    # -------------------------------------------------- Frequency - Basic
    def set_freq(self, value: float, unit: UnitFreq = UnitFreq.HZ) -> None:
        """Set CW carrier frequency."""
        self.specs.validate_frequency(value, unit)
        self.write(f":FREQuency:CW {value} {unit.value}")

    def get_freq(self) -> float:
        """Return CW carrier frequency in Hz."""
        return self._query_float(":FREQuency:CW?")

    def set_freq_mode(self, mode: ModeFreq = ModeFreq.CW) -> None:
        """Set frequency mode: CW or LIST."""
        if not isinstance(mode, ModeFreq):
            raise TypeError("mode must be ModeFreq")
        self.write(f":FREQuency:MODE {mode.value}")

    def get_freq_mode(self) -> str:
        """Return frequency mode."""
        return self.query(":FREQuency:MODE?").strip()

    # -------------------------------------------------- Frequency - Sweep
    def set_freq_start(self, value: float, unit: UnitFreq = UnitFreq.HZ) -> None:
        """Set sweep start frequency."""
        self.specs.validate_frequency(value, unit)
        self.write(f":FREQuency:STARt {value} {unit.value}")

    def get_freq_start(self) -> float:
        """Return sweep start frequency in Hz."""
        return self._query_float(":FREQuency:STARt?")

    def set_freq_stop(self, value: float, unit: UnitFreq = UnitFreq.HZ) -> None:
        """Set sweep stop frequency."""
        self.specs.validate_frequency(value, unit)
        self.write(f":FREQuency:STOP {value} {unit.value}")

    def get_freq_stop(self) -> float:
        """Return sweep stop frequency in Hz."""
        return self._query_float(":FREQuency:STOP?")
    
    # -------------------------------------------------- Frequency - Misc Queries 
    def get_freq_multiplier(self) -> int:
        """Return display frequency multiplier."""
        return int(float(self.query(":FREQuency:MULTiplier?").strip()))

    def get_freq_offset(self) -> float:
        """Return display frequency offset in Hz."""
        return self._query_float(":FREQuency:OFFSet?")

    def get_freq_reference(self) -> float:
        """Return frequency reference value in Hz."""
        return self._query_float(":FREQuency:REFerence?")

    def get_freq_reference_state(self) -> bool:
        """Return whether frequency reference mode is enabled."""
        return self._query_bool(":FREQuency:REFerence:STATe?")

    def get_phase_adjust(self) -> float:
        """Return phase adjustment in radians."""
        return self._query_float(":PHASe:ADJust?")

    def get_ref_osc_source(self) -> str:
        """Return reference oscillator source: INT or EXT."""
        return self.query(":ROSCillator:SOURce?").strip()

    def get_ref_osc_auto(self) -> bool:
        """Return whether automatic reference oscillator selection is enabled."""
        return self._query_bool(":ROSCillator:SOURce:AUTO?")
  
    # -------------------------------------------------- Frequency - Modulation Subsystem     
    def get_fm_state(self, path: int = 1) -> bool: 
        """Return whether the FM path (1 or 2) is enabled or not"""
        path = self._path(path)
        return self._query_bool(f":FM{path}:STATe?")

    def set_fm_state(self, state: bool, path: int = 1) -> None:
        """Enable or disable the FM path (1 or 2)."""
        path = self._path(path)
        state = self._bool(state)
        self.write(f":FM{path}:STATe {state}")
    
    def get_fm_source(self, path: int = 1) -> str:
        """Return FM source for path 1 or 2: INT, EXT1, or EXT2."""
        path = self._path(path)
        return self.query(f":FM{path}:SOURce?").strip()

    def set_fm_source(self, source: ModeSource, path: int = 1) -> None:
        """Set FM source for path 1 or 2: INT, EXT1, or EXT2."""
        path = self._path(path)

        if not isinstance(source, ModeSource):
            raise TypeError("source must be ModeSource")

        self.write(f":FM{path}:SOURce {source.value}")

    def get_fm_coupling(self, path: int = 1, external: int = 1) -> str:
        """Return AC/DC coupling for FM external input 1 or 2."""
        path = self._path(path)
        external = self._path(external)
        return self.query(f":FM{path}:EXTernal{external}:COUPling?").strip()


    def set_fm_coupling( self, coupling: ModeCoupling, 
                               path: int = 1,
                               external: int = 1 ) -> None:
        """Set AC/DC coupling for FM external input 1 or 2."""
        path = self._path(path)
        external = self._path(external)

        if not isinstance(coupling, ModeCoupling):
            raise TypeError("coupling must be ModeCoupling")

        self.write(f":FM{path}:EXTernal{external}:COUPling {coupling.value}")
    
    def get_fm_deviation(self, path: int = 1) -> float:
        """Return FM peak deviation in Hz for path 1 or 2."""
        path = self._path(path)
        return self._query_float(f":FM{path}:DEViation?")

    def set_fm_deviation(self, deviation: float, unit: UnitFreq = UnitFreq.HZ, path: int = 1) -> None:
        """
        Set FM peak deviation. Deviation Maximum is Carrier-frequency dependent:
        - 250 Khz - 250 MHz : 10    MHz   
        - 250 MHz - 500 MHz : 5     MHz
        - 500 MHz - 1 GHz   : 10    MHz
        - 1 GHz   - 2 GHz   : 20    MHz
        - 2 GHz   - 4 GHz   : 40    MHz
        This method does not validate the carrier-dependent FM deviation limit.
        """
        path = self._path(path)
        self.write(f":FM{path}:DEViation {deviation} {unit.value}")

    def set_fm_rate(self, rate: float, unit: UnitFreq = UnitFreq.HZ, path: int = 1) -> None:
        """
        Set internal FM modulation rate.

        Important:
        For normal FM rate:
        - sine waveform: 0.1 Hz to 50 kHz
        - square/ramp/triangle waveform: 0.1 Hz to 10 kHz

        Swept-sine and dual-sine FM rate settings can also use 0.1 Hz to 50 kHz.

        This method does not validate the waveform-dependent FM rate limit.
        """ 
        path = self._path(path)
        self.write(f":FM{path}:INTernal:FREQuency {rate} {unit.value}")

    def get_fm_rate(self, path: int = 1) -> float:
        """Return internal FM modulation rate in Hz for path 1 or 2."""
        path = self._path(path)
        return self._query_float(f":FM{path}:INTernal:FREQuency?")

    def get_fm_shape(self, path: int = 1) -> str:
        """Return internal FM modulation waveform shape for path 1 or 2."""
        path = self._path(path)
        return self.query(f":FM{path}:INTernal:FUNCtion:SHAPe?").strip()

    def set_fm_shape(self, shape: ModeShape, path: int = 1) -> None:
        """Set internal FM modulation waveform shape for path 1 or 2."""
        path = self._path(path)

        if not isinstance(shape, ModeShape):
            raise TypeError("shape must be ModeShape")

        self.write(f":FM{path}:INTernal:FUNCtion:SHAPe {shape.value}")

    # -------------------------------------------------- Amplitude - Modulation
    def get_am_state(self, path: int = 1) -> bool:
        """Return whether AM path is enabled."""
        path = self._path(path)
        return self._query_bool(f":AM{path}:STATe?")

    def set_am_state(self, state: bool, path: int = 1) -> None:
        """Enable or disable AM path."""
        path = self._path(path)
        state = self._bool(state)
        self.write(f":AM{path}:STATe {state}")

    def get_am_source(self, path: int = 1) -> str:
        """Return AM source for path 1 or 2."""
        path = self._path(path)
        return self.query(f":AM{path}:SOURce?").strip()

    def set_am_source(self, source: ModeSource, path: int = 1) -> None:
        """Set AM source for path 1 or 2."""
        path = self._path(path)
        if not isinstance(source, ModeSource):
            raise TypeError("source must be ModeSource")
        self.write(f":AM{path}:SOURce {source.value}")

    def get_am_depth(self, path: int = 1) -> float:
        """Return AM depth in percent."""
        path = self._path(path)
        return self._query_float(f":AM{path}:DEPTh?")

    def set_am_depth(self, depth_percent: float, path: int = 1) -> None:
        """Set AM depth in percent. (0.1 - 100 %)"""
        path = self._path(path)
        self.specs.validate_am_depth(depth_percent)
        self.write(f":AM{path}:DEPTh {depth_percent} PCT")

    def get_am_rate(self, path: int = 1) -> float:
        """Return internal AM modulation rate in Hz."""
        path = self._path(path)
        return self._query_float(f":AM{path}:INTernal:FREQuency?")

    def set_am_rate( self,  rate: float,
                            unit: UnitFreq = UnitFreq.HZ,
                            path: int = 1 ) -> None:
        """
        Set internal AM modulation rate.

        Important:
        For normal AM rate:
        - sine waveform: 0.1 Hz to 50 kHz
        - square/ramp/triangle waveform: 0.1 Hz to 10 kHz

        Swept-sine and dual-sine AM rate settings can also use 0.1 Hz to 50 kHz.

        This method does not validate the waveform-dependent AM rate limit.
        """
        path = self._path(path)
        self.write(f":AM{path}:INTernal:FREQuency {rate} {unit.value}")

    def get_am_shape(self, path: int = 1) -> str:
        """Return internal AM waveform shape."""
        path = self._path(path)
        return self.query(f":AM{path}:INTernal:FUNCtion:SHAPe?").strip()

    def set_am_shape(self, shape: ModeShape, path: int = 1) -> None:
        """Set internal AM waveform shape."""
        path = self._path(path)
        if not isinstance(shape, ModeShape):
            raise TypeError("shape must be ModeShape")
        self.write(f":AM{path}:INTernal:FUNCtion:SHAPe {shape.value}")

    # -------------------------------------------------- Power - Basic
    def set_power(self, value: float, unit: UnitPower = UnitPower.DBM) -> None:
        """Set RF output power level. Only dBm works right now (TODO Others)"""
        self.specs.validate_power(value, unit)
        self.write(f":POWer:AMPLitude {value} {unit.value}")

    def get_power(self) -> float:
        """Return RF output power level in the active power unit."""
        return self._query_float(":POWer:AMPLitude?")

    def set_power_unit(self, unit: UnitPower = UnitPower.DBM) -> None:
        """Set RF output power unit."""
        if not isinstance(unit, UnitPower):
            raise TypeError("unit must be UnitPower")
        self.write(f":UNIT:POWer {unit.value}")

    def set_power_mode(self, mode: ModePower = ModePower.FIX) -> None:
        """Set RF output power mode: FIX or LIST."""
        if not isinstance(mode, ModePower):
            raise TypeError("mode must be ModePower")
        self.write(f":POWer:MODE {mode.value}")

    def get_power_mode(self) -> str:
        """Return RF output power mode."""
        return self.query(":POWer:MODE?").strip()

    # -------------------------------------------------- Power - SWEEP 
    def set_power_start(self, value: float, unit: UnitPower = UnitPower.DBM) -> None:
        """Set RF output start power for power sweep."""
        self.specs.validate_power(value, unit)
        self.write(f":POWer:STARt {value} {unit.value}")

    def get_power_start(self) -> float:
        """Return RF output start power."""
        return self._query_float(":POWer:STARt?")

    def set_power_stop(self, value: float, unit: UnitPower = UnitPower.DBM) -> None:
        """Set RF output stop power for power sweep."""
        self.specs.validate_power(value, unit)
        self.write(f":POWer:STOP {value} {unit.value}")

    def get_power_stop(self) -> float:
        """Return RF output stop power."""
        return self._query_float(":POWer:STOP?")

    # -------------------------------------------------- Power - ALC 
    def set_power_alc(self, state: bool) -> None:
        """Turn automatic level control on or off."""
        self.write(f":POWer:ALC:STATe {self._bool(state)}")

    def get_power_alc(self) -> bool:
        """Return whether automatic level control is enabled."""
        return self._query_bool(":POWer:ALC:STATe?")
    
    # -------------------------------------------------- Power - MISC
    def get_power_reference(self) -> float:
        """Return RF output power reference level."""
        return self._query_float(":POWer:REFerence?")

    def get_power_reference_state(self) -> bool:
        """Return whether RF output power reference mode is enabled."""
        return self._query_bool(":POWer:REFerence:STATe?")
    
    # -------------------------------------------------- Trigger - Basic
    def trigger_bus(self):
        ''' Works only if Bus Triggering is the type of event selected'''
        self.write("*TRG")

    def trigger_immediate(self) -> None:
        """Send an immediate trigger to a sweep waiting for trigger."""
        self.write(":TRIGger:SEQuence:IMMediate")

    def set_trigger_source(self, source: ModeTrig) -> None:
        """Set sweep trigger source."""
        if not isinstance(source, ModeTrig):
            raise TypeError("source must be ModeTrig")
        self.write(f":TRIGger:SEQuence:SOURce {source.value}")

    def get_trigger_source(self) -> str:
        """Return sweep trigger source."""
        return self.query(":TRIGger:SEQuence:SOURce?").strip()
    
    # -------------------------------------------------- Trigger - Technical
    def set_trigger_slope(self, slope: ModeSlope) -> None:
        """Set external trigger input slope."""
        if not isinstance(slope, ModeSlope):
            raise TypeError("slope must be ModeSlope")
        self.write(f":TRIGger:SEQuence:SLOPe {slope.value}")

    def get_trigger_slope(self) -> str:
        """Return external trigger input slope."""
        return self.query(":TRIGger:SEQuence:SLOPe?").strip()

    def set_trigger_output_polarity(self, slope: ModeSlope) -> None:
        """Set trigger output TTL polarity."""
        if not isinstance(slope, ModeSlope):
            raise TypeError("slope must be ModeSlope")
        self.write(f":TRIGger:OUTPut:POLarity {slope.value}")

    def get_trigger_output_polarity(self) -> str:
        """Return trigger output TTL polarity."""
        return self.query(":TRIGger:OUTPut:POLarity?").strip()
    
    # -------------------------------------------------- Errors
    def get_error(self) -> str:
        return self.query(":SYSTem:ERRor?").strip()

    def get_all_errors(self, max_errors: int = 20) -> List[str]:
        errors = []
        for _ in range(max_errors):
            err = self.get_error()
            errors.append(err)
            if err.startswith("0") or "No error" in err:
                break
        return errors
    
    # -------------------------------------------------- State - Saving / Recalling
    def _state_register(self, reg: int, seq: int) -> Tuple[int, int]:
        ''' Validate SCPI status register and sequence indices.'''
        if not (0 <= reg <= 99):
            raise ValueError("reg must be between 0 and 99")
        if not (0 <= seq <= 9):
            raise ValueError("seq must be between 0 and 9")
        return reg, seq
    
    def state_save(self, reg: int, seq: int) -> None:
        """Save the current instrument state to USER/STATE register reg in sequence seq."""
        reg, seq = self._state_register(reg, seq)
        self.write(f"*SAV {reg},{seq}")

    def state_recall(self, reg: int, seq: int) -> None:
        """Recall a saved instrument state from USER/STATE register reg in sequence seq."""
        reg, seq = self._state_register(reg, seq)
        self.write(f"*RCL {reg},{seq}")

    # -------------------------------------------------- Sweeping - Basic
    def sweep_abort(self) -> None:
        self.write(":ABORt")

    def sweep_initiate_single(self) -> None:
        """Initiate one sweep if frequency or power sweep is enabled."""
        self.write(":INITiate:IMMediate:ALL")

    def set_sweep_type(self, sweep_type: ModeSweep = ModeSweep.STEP) -> None:
        if not isinstance(sweep_type, ModeSweep):
            raise TypeError("sweep_type must be ModeSweep")
        self.write(f":LIST:TYPE {sweep_type.value}")

    def set_sweep_continuous(self, state: bool) -> None:
        """Enable or disable continuous sweep mode."""
        state = self._bool(state)
        self.write(f":INITiate:CONTinuous:ALL {state}")

    def get_sweep_continuous(self) -> bool:
        """Return whether continuous sweep mode is enabled."""
        return self._query_bool(":INITiate:CONTinuous:ALL?")

    def set_sweep_dwell(self, seconds: float) -> None:
        """Set dwell time per sweep point in seconds."""
        self.specs.validate_sweep_dwell(seconds)
        self.write(f":SWEep:DWELl {seconds}")

    def get_sweep_dwell(self) -> float:
        """Return dwell time per sweep point in seconds."""
        return self._query_float(":SWEep:DWELl?")

    def set_sweep_points(self, points: int) -> None:
        """Set number of points in a step sweep."""
        if not isinstance(points, int):
            raise TypeError("points must be int")
        self.specs.validate_sweep_points(points)
        self.write(f":SWEep:POINts {points}")

    def get_sweep_points(self) -> int:
        """Return number of points in a step sweep."""
        return int(float(self.query(":SWEep:POINts?").strip()))

    # -------------------------------------------------- Modulation - Internal
    def modulate_internal_freq( self,    carrier: float,
                                carrier_unit: UnitFreq = UnitFreq.HZ,
                                power_dbm: float = 0.0,         # 1mW
                                deviation: float = 50.0,
                                deviation_unit: UnitFreq = UnitFreq.KHZ,
                                rate: float = 10.0,
                                rate_unit: UnitFreq = UnitFreq.KHZ,
                                path: int = 1,
                                shape: ModeShape = ModeShape.SINE ) -> None:
        """ Reset Then Configure a basic internally frequency-modulated RF signal."""
        
        # --- Prepare
        path = self._path(path)
        self.reset()
        self.set_output_rf(False)           
        self.set_output_modulation(False)

        # --- Carrier setup
        self.set_freq(carrier, carrier_unit)
        self.set_power(power_dbm, UnitPower.DBM) 

        # --- FM setup
        self.set_fm_shape(shape, path)
        self.set_fm_source(ModeSource.INT, path)
        self.set_fm_deviation(deviation, deviation_unit, path)
        self.set_fm_rate(rate, rate_unit, path)
        self.set_fm_state(True, path)

        # --- Enable output last
        self.set_output_modulation(True)
        self.set_output_rf(True)

    def modulate_internal_am( self,  carrier: float,
                            carrier_unit: UnitFreq = UnitFreq.HZ,
                            power_dbm: float = 0.0,
                            depth_percent: float = 50.0,
                            rate: float = 1.0,
                            rate_unit: UnitFreq = UnitFreq.KHZ,
                            path: int = 1,
                            shape: ModeShape = ModeShape.SINE ) -> None:
        """Configure a basic internally amplitude-modulated RF signal."""

        # --- Prepare
        path = self._path(path)
        if not (0.0 <= depth_percent <= 100.0):
            raise ValueError("depth_percent must be between 0 and 100")
        self.reset()
        self.set_output_rf(False)
        self.set_output_modulation(False)

        # --- Carrier setup
        self.set_freq(carrier, carrier_unit)
        self.set_power(power_dbm, UnitPower.DBM)

        # --- AM setup
        self.set_am_shape(shape, path)
        self.set_am_source(ModeSource.INT, path)
        self.set_am_depth(depth_percent, path)
        self.set_am_rate(rate, rate_unit, path)
        self.set_am_state(True, path)

        # --- Enable output last
        self.set_output_modulation(True)
        self.set_output_rf(True)

    # -------------------------------------------------- SWEEP - Control
    def _safe_sweep_power_dbm(self, value: float) -> None:
        if not (-136.0 <= value <= 7.0):
            raise ValueError("Sweep power must be between -136 and +7 dBm for safe high-level use")

    def sweep_step_frequency_config(self, start: float,
                                        stop: float,
                                        points: int,
                                        dwell_s: float,
                                        unit: UnitFreq = UnitFreq.KHZ,
                                        power_dbm: float  = 0.0 ,
                                        continuous: bool = False ) -> None:
        
        """Configure a step frequency sweep."""
        self.reset()
        self.set_output_rf(False)
        self.set_sweep_type(ModeSweep.STEP)

        if power_dbm is not None:
            self._safe_sweep_power_dbm(power_dbm)
            self.set_power(power_dbm, UnitPower.DBM)

        self.set_freq_start(start, unit)
        self.set_freq_stop(stop, unit)

        self.set_sweep_points(points)
        self.set_sweep_dwell(dwell_s)

        self.set_freq_mode(ModeFreq.LIST)
        self.set_sweep_continuous(continuous)

        self.set_output_rf(True)

    def sweep_step_power_config( self,    start_power: float,
                                        stop_power: float,
                                        points: int,
                                        dwell_s: float,
                                        unit: UnitPower = UnitPower.DBM,
                                        freq: float  = 0.0,
                                        freq_unit: UnitFreq = UnitFreq.HZ,
                                        continuous: bool = False ) -> None:
        """Configure a step power sweep."""
        self.reset()
        self.set_output_rf(False)
        self.set_sweep_type(ModeSweep.STEP)

        if freq is not None:
            self.set_freq(freq, freq_unit)

        if unit == UnitPower.DBM:
            self._safe_sweep_power_dbm(start_power)
            self._safe_sweep_power_dbm(stop_power)

        self.set_power_start(start_power, unit)
        self.set_power_stop(stop_power, unit)

        self.set_sweep_points(points)
        self.set_sweep_dwell(dwell_s)

        self.set_power_mode(ModePower.LIST)
        self.set_sweep_continuous(continuous)

        self.set_output_rf(True)

    def sweep_step_frequency_power_config( self,  start_freq: float,
                                                stop_freq: float,
                                                start_power: float,
                                                stop_power: float,
                                                points: int,
                                                dwell_s: float,
                                                freq_unit: UnitFreq = UnitFreq.HZ,
                                                power_unit: UnitPower = UnitPower.DBM,
                                                continuous: bool = False ) -> None:
        """Configure a step sweep of frequency and power together."""
        self.reset()
        self.set_output_rf(False)
        self.set_sweep_type(ModeSweep.STEP)

        if power_unit == UnitPower.DBM:
            self._safe_sweep_power_dbm(start_power)
            self._safe_sweep_power_dbm(stop_power)

        self.set_freq_start(start_freq, freq_unit)
        self.set_freq_stop(stop_freq, freq_unit)

        self.set_power_start(start_power, power_unit)
        self.set_power_stop(stop_power, power_unit)

        self.set_sweep_points(points)
        self.set_sweep_dwell(dwell_s)

        self.set_freq_mode(ModeFreq.LIST)
        self.set_power_mode(ModePower.LIST)
        self.set_sweep_continuous(continuous)

        self.set_output_rf(True)


    # -------------------------------------------------- SWEEP - Control 
    def sweep_start_immediate(self) -> None:
        self.set_trigger_source(ModeTrig.IMM)
        self.sweep_initiate_single()

    def sweep_arm_external(self, slope: ModeSlope = ModeSlope.POS) -> None:
        self.set_trigger_source(ModeTrig.EXT)
        self.set_trigger_slope(slope)
        self.sweep_initiate_single()

    # -------------------------------------------------- Modulation - EXTERNAL
    # TODO

    # -------------------------------------------------- SWEEP - LIST 
    # TODO For That we need the LIST SUBSYSTEM FIRST (LATER)

    # -------------------------------------------------- LIST Subsystem
    # TODO 

    # -------------------------------------------------- LF System
    # TODO (No Lf output in our device  - PDF 109 in Programmer's Guide) 

    # -------------------------------------------------- Modulation - EXTERNAL
    # TODO

    # -------------------------------------------------- Phase Modulation System 
    # TODO 

    # -------------------------------------------------- Puls Modulation System 
    # TODO 


    # -------------------------------------------------- Memory Subsystem
    # TODO

