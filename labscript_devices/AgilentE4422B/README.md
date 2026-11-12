# Agilent E4422B Labscript Device

Minimal `labscript` / BLACS device support for the Agilent E4422B ESG-A RF signal generator.

This device currently supports basic static RF output control:

- Carrier frequency
- RF output power
- RF output ON/OFF through BLACS
- Manual BLACS readback
- Static shot programming through HDF5

The lower-level SCPI wrapper also contains early support for AM, FM, sweep, trigger, and status/error commands, but not all of these are fully wired into BLACS or labscript yet.

---

## Device

Instrument:

- Manufacturer: Agilent
- Model: E4422B
- Series: ESG-A
- Frequency range: `250 kHz` to `4 GHz`
- Power range: `-136 dBm` to `+7 dBm`

---


## Example Labscript Usage

```python

# ------------------------ Connection table 
AgilentE4422B( name="rfgen", GPIB_address="GPIB0::5")                       # If used with GPIB port 
AgilentE4422B( name="rfgen", GPIB_address="ADAP::IP_ADRESS::GPIB_ADRESS")   # If used with Prologix like Adapter  
AgilentE4422BRFOutput("rf", parent_device=rfgen, connection="rf")

# ------------------------ Experiments script

# --- Set Freq
rf.setfreq_mhz(100)     # in MHz
rf.setfreq_khz(250)     # in KHz

# --- Set Amp
rf.setamp_dbm(-20) 

# --- Set Rf Output On/Off
rf.set_output_rf(True)




