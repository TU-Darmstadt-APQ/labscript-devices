# Subsystems (TODO EDIT THIS README)

## Frequency Subsystem

### Set Freq

- RF ON/OFF Key Must Be ON 
- Press Frequency:  then Eg. 700 then MHZ
- Freq is the active Function until you press another function
- Incr or Decrease Freq with up and down Arrow (Increment can be adjusted with Incr Set key)
- There is a front Panel Knob 


## Amplitude Subsystem

### Set Amplitude RF OutPowerLevel

- Preset to get factory-defined instrument state (-135 dBM)
- RF ON/OFF KEY Must Be ON
- Press Amplitude: then Eg. -20 then MHZ 
- Like Freq , use arrows and Knob



# Modulation

## Amplitude Modulation

Example Modulation:  carrier freq  1340 kHz , power level  0 dBm , AM depth 90% , AM rate 10 kHz

NOTE Preset First (Default RF OFF , I )

- Step 1 : Set Carrier                  -> use : Set Freq 1340 khz (NOTE !!! But dont Turn RF ON !!!)
- Step 2 : Set Power Level              -> use:  Set Amplitude RF OutPowerLevel 0 dbm
- Step 3 : set AM Depth                 -> Press AM depth Key, 90 then %
- Step 4 : set AM Depth                 -> press Am rate  Key, 10 then khz
- Step 5 : Turn On Amplitude Modulation -> Press MOD ON, AM ON ,then RF ON

!!!!!!! NOTE: Do the steps, BUT NOTE AM ON and RF ON are pressed only at the end !!!!!!!!!

## Frequency Modulation

Same like for the Amplitude Modulation mit FM

SO FM/ΦM key -> FM DEV <num> -> FM RATE <num> -> FM ON -> RF ON

## Modulation Parameters Summary 
- state                 → On / OFF          
- carrier frequency     → main RF frequency 
- FM deviation          → max frequency swing
- FM source             → INT / EXT1 / EXT2
- FM coupling           → AC / DC in case source EXT (*RST Value: DC)
- FM internal waveform  → sine, triangle, square, etc.
- FM internal rate      → modulation speed


## Also Possible is : 
- Coupling Deviations between Path 1 and 2 (Programming Manual : Frequency Modulation Deviation Coupling)
- Dual-sine tone 2 / swept-sine stop rate can be configured (Programming Manual: Internal Frequency Modulation Alternate Frequency)
- Dual-sine tone 2 contribution to total FM depth can be configured in percent (Programming Manual: Internal Frequency Modulation Alternate Frequency Amplitude)
- Swept-sine sweep duration can be configured (Programming Manual: Internal Frequency Modulation Sweep Time)
- Swept-sine trigger source can be configured: immediate, bus, external, or key (Programming Manual: Internal Frequency Modulation Sweep Trigger)

# Sweeping

## Step Sweep

Example : Freq Range 525Mhz 600Mhz , Power Level -20dbm, dwelltime  500 ms

NOTE: Don't forget to Preset 

Step1 : configuring step sweep : Press KEy Sweep/List -> Toggle Key Sweeep Type List/Step to STep

- Edit Freq Start and Freq Stop (Key Freq Start and stop ) 
- Edit Power level  (Key Ampl Start and stop)
- Edit Num of sweeep points (key: #0Points)
- Edit Dwell Soft KEy (Key Step Dwell) 

Step2: Turn On Continuous Sweep (NOTE continuous sweeping in this example)

- Press Return 
- Press Key Sweep -> choose from (Freq , Ampl or Frq/Ampl) -> Here choose Freq/ AMpl
- Press Sweeep Repeat single/Cont and toggle "cont"
- Last Press RF ON

## List Sweep

TODO Needed ? NOTE Very flexible sweeping

--- 

# NOTE 

- Puls Modulation Possible only on internal and EXT2

- Some ScpI `:SYSTem:LANGuage?` , `:SYSTem:VERSion?` 

- A Path is one configuration object per modulation channel.
- In this RF generator there are 2 paths per Channels 
    This Means 
    - AM has 2 paths: AM1, AM2
    - FM has 2 paths: FM1, FM2
    - ΦM has 2 paths: ΦM1, ΦM2
    - ...

- ALTERNATE : is a second moulation parameter that is needed in special cases like DUALsine or SWEPTsine