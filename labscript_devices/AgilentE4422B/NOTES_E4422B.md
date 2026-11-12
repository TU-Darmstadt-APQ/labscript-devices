# Notes

- Pulse modulation is possible only on internal and EXT2.

- A path is one configuration object per modulation channel.
- In this RF generator there are 2 paths per channel:
  - AM has 2 paths: AM1, AM2
  - FM has 2 paths: FM1, FM2
  - ΦM has 2 paths: ΦM1, ΦM2
  - ...

- ALTERNATE is a second modulation parameter used in special cases like dual-sine or swept-sine.



# Possible on the device 

- Coupling deviations between Path 1 and Path 2  
  → Programming Manual: Frequency Modulation Deviation Coupling

- Dual-sine tone 2 / swept-sine stop rate can be configured  
  → Programming Manual: Internal Frequency Modulation Alternate Frequency

- Dual-sine tone 2 contribution to total FM depth can be configured in percent  
  → Programming Manual: Internal Frequency Modulation Alternate Frequency Amplitude

- Swept-sine sweep duration can be configured  
  → Programming Manual: Internal Frequency Modulation Sweep Time

- Swept-sine trigger source can be configured: immediate, bus, external, or key  
  → Programming Manual: Internal Frequency Modulation Sweep Trigger


# Subsystems 

## Frequency Subsystem

### Set Freq

- Press Frequency, then e.g. `700 MHz`.
- Freq is the active function until you press another function.
- Increase or decrease freq with up and down arrows.
  - Increment can be adjusted with the Incr Set key.
- There is a front panel knob.
- RF ON/OFF can be turned ON after setting the frequency.

## Amplitude Subsystem

### Set Amplitude RF OutPowerLevel

- Preset gives the factory-defined instrument state. Default amplitude is `-135 dBm`.
- Press Amplitude, then e.g. `-20 dBm`.
- Like freq, use arrows and knob.
- RF ON/OFF can be turned ON after setting the amplitude.


# Modulation

## Amplitude Modulation

Example modulation: carrier freq `1340 kHz`, power level `0 dBm`, AM depth `90%`, AM rate `10 kHz`.

NOTE: Preset first. Default RF is OFF.

- Step 1: Set Carrier  
  → use: Set Freq `1340 kHz`  
  → NOTE: Do not turn RF ON yet.

- Step 2: Set Power Level  
  → use: Set Amplitude RF OutPowerLevel `0 dBm`.

- Step 3: Set AM Depth  
  → press AM Depth key, `90`, then `%`.

- Step 4: Set AM Rate  
  → press AM Rate key, `10`, then `kHz`.

- Step 5: Turn On Amplitude Modulation  
  → press AM ON, then RF ON.

NOTE: Do the steps first. AM ON and RF ON are pressed only at the end.

## Frequency Modulation

Same like for amplitude modulation, but with FM.

FM/ΦM key → FM DEV `<num>` → FM RATE `<num>` → FM ON → RF ON.

## Modulation Parameters Summary

- state → ON / OFF
- carrier frequency → main RF frequency
- FM deviation → max frequency swing
- FM source → INT / EXT1 / EXT2
- FM coupling → AC / DC in case source EXT (`*RST` value: DC)
- FM internal waveform → sine, triangle, square, etc.
- FM internal rate → modulation speed


 ----
 
# Sweeping

## Step Sweep

Example: freq range `525 MHz` to `600 MHz`, power level `-20 dBm`, dwell time `500 ms`.

NOTE: Do not forget to Preset.

Step 1: Configure step sweep  
Press Sweep/List → toggle Sweep Type List/Step to Step.

- Edit Freq Start and Freq Stop.
- Edit Power Level.
- Edit number of sweep points.
- Edit dwell time with Step Dwell.

Step 2: Turn on continuous sweep  
NOTE: continuous sweeping is used in this example.

- Press Return.
- Press Sweep → choose Freq, Ampl, or Freq/Ampl.
- Here choose Freq/Ampl.
- Press Sweep Repeat Single/Cont and toggle to Cont.
- Last press RF ON.

## List Sweep

TODO: Needed ?  
NOTE: Very flexible sweeping.

