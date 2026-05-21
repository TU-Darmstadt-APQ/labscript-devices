from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional, Union

# ------------------------------------------------------ Typing
Number = Union[int , float]

# ------------------------------------------------------ Units
class UnitFreq(str, Enum):
    HZ  = "Hz"
    KHZ = "kHz"
    MHZ = "MHz"
    GHZ = "GHz"

factors_freq_to_hz = { UnitFreq.HZ: 1.0,   
            UnitFreq.KHZ: 1e3,
            UnitFreq.MHZ: 1e6,
            UnitFreq.GHZ: 1e9 }

def freq_to_hz(value: Number, unit: UnitFreq) -> float:
    if not isinstance(unit, UnitFreq):
        raise TypeError("unit must be UnitFreq")

    return float(value) * factors_freq_to_hz[unit]


class UnitPower(str, Enum):
    """RF output power units used by the signal generator."""
    DBM  = "DBM"   # Power in dBm, referenced to 1 mW.
    # TODO Followings Need Validations
    # DBUV = "DBUV"  # Voltage level in dBµV, referenced to 1 microvolt.
    # V    = "V"     # RMS voltage at the RF output.
    # VEMF = "VEMF"  # EMF voltage; source voltage before 50 ohm load division.

# ------------------------------------------------------ Some Basics
@dataclass(frozen=True)
class DeviceSpecs:
    manufact: str
    model: str
    series: str


@dataclass(frozen=True)
class NumericSpec:
    minimum: float
    maximum: float
    unit: Optional[str] = None

    def validate(self, value: Number) -> None:

        # --- Validate that is a Number 
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("value must be int or float")

        # --- Validate the range
        if not self.minimum <= value <= self.maximum:
            unit = " {}".format(self.unit) if self.unit else ""
            raise ValueError(
                "Value {}{} is outside valid range: {}{} to {}{}".format(
                    value, unit, self.minimum, unit, self.maximum, unit)
            )
        
# ------------------------------------------------------ The RF Generator Class
@dataclass(frozen=True)
class RFGeneratorSpecs(DeviceSpecs):
    ''' Datasheet that can take care of validations as well. '''
    frequency: NumericSpec
    power_dbm: NumericSpec

    sweep_points: Optional[NumericSpec] = None
    sweep_dwell_s: Optional[NumericSpec] = None

    am_depth_percent: Optional[NumericSpec] = None


    def validate_frequency(self, value: Number, unit: UnitFreq = UnitFreq.HZ) -> None:
        self.frequency.validate(freq_to_hz(value, unit))

    def validate_power(self, value: Number, unit: UnitPower = UnitPower.DBM) -> None:
        if not isinstance(unit, UnitPower):
            raise TypeError("unit must be UnitPower")

        if unit is UnitPower.DBM and self.power_dbm is not None:
            self.power_dbm.validate(value)

    def validate_sweep_points(self, points: Number) -> None:
        if self.sweep_points is not None:
            self.sweep_points.validate(points)

    def validate_sweep_dwell(self, seconds: Number) -> None:
        if self.sweep_dwell_s is not None:
            self.sweep_dwell_s.validate(seconds)

    def validate_am_depth(self, depth_percent: Number) -> None:
        if self.am_depth_percent is not None:
            self.am_depth_percent.validate(depth_percent)


# ------------------------------------------------------ STATS 
@dataclass
class RFGeneratorStats:
    freq_mhz : float
    power_dbm : float 
    rf_on : Optional[bool]