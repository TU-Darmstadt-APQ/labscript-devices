# HV Series High Voltage Source

It adds initial support for the Stahl Electronics HV-200-8 and HV-250-8 
multichannel voltage sources to the labscript suite.

Commands are sent via a standard serial interface with a standard baud 
rate of 9600.

---
## Timing Strategy

Since the HV devices have no internal clock and no buffering, 
time-sensitive operations (e.g., updating voltages across a sequence) 
are implemented using `time.sleep(t)` in a background thread, opened in 
`transition_to_buffered`. After the whole sequence is done, the thread is 
closed in `transition_to_manual`.

While this is a naive and dirty approach, it currently works to avoid 
blocking the main thread. However, we acknowledge that this is not ideal,
and we welcome proposals for a cleaner, event-driven timing model.

---
## GUI small extension

A "Send to Device" button that:
- Collects all entered voltages 
- Queues them into the worker process 
- Sends commands serially to the device

A "Check Remote Values" button that:
- Checks remote values 
- Provides choice to program device with values from the front panel or remote values 
- After clicking `apply`, the user needs to click `send to device` to actually reprogram the device with values from the front panel

---
## Emulator

This [emulator](testing/emulateSerPort.py) simulates the behavior of the device. 
It allows for testing with BLACS. When started, the emulator creates 
a virtual serial port that behaves like a real device. Programs can 
connect to this port and communicate with it as if it were the real device.

To launch the emulator:

```bash 
python3 -m HV_stahl.testing.emulateSerPort  
```

You’ll see output like: use: `/dev/pts/5`

Use that port when connecting in `connection_table.py`.

---
## Current implementation

The current implementation consists of HV-200 and HV-250 as models. 
We are modeling this modular support similar to how **NI_DAQmx** handles 
multiple device types.

Other devices in the same family are expected to have similar 
interfaces but may differ in:
- Voltage ranges per channel 
- Number of channels 
- Supported commands 
- etc.

The plan is to extend the current implementation by using 
subclassing and extending the configurations in [capabilities.json](models/capabilities.json).

In detail:

To add a new device, the user should:

- Define the new device model in the capabilities.json file 
- Create a corresponding .py file (by copying an existing model as a template)
- Adjust the class and the model names in the .py file to match the new entry in the JSON file

Note: The class name must exactly match the model name specified in capabilities.json.****