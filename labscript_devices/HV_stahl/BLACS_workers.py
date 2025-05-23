from blacs.tab_base_classes import Worker
from labscript import LabscriptError
import serial
from user_devices.logger_config import logger
import time


class HV_Worker(Worker):
    def init(self):
        """Initialises communication with the device. When BLACS (re)starts"""
        
        try:
            self.connection = serial.Serial(self.port, self.baud_rate, timeout=1)
            logger.info(f"Connected to HV_ on {self.port} with baud rate {self.baud_rate}")
            
            # Identify the device:
            self.send_to_HV("IDN\r")
            device_info = self.receive_from_HV().split()
            logger.debug(f"Receiving identification HV-200: {device_info}")
            if len(device_info) >= 4:
                self.device_serial = device_info[0]
                self.device_voltage_range = device_info[1]
                self.device_channels_number= device_info[2]
                self.device_output_type = device_info[3]
                logger.info(f"Device Serial: {self.device_serial}, Voltage Range: {self.device_voltage_range}, Channels: {self.device_channels_number}, Output Type: {self.device_output_type}")
            else:
                raise RuntimeError("Failed to parse the device's identification string. ")
                
        except serial.SerialException as e:
            raise RuntimeError(f"Serial connection error: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during HV_ worker initialization: {e}")
        
        
    def shutdown(self):
        # Should be done when Blacs is closed
        self.connection.close()
    
    def scale_to_range(self, normalized_value, max_range):
        """Convert a normalized value (0 to 1) to the specified range (-max_range to +max_range)"""
        max_range = float(max_range)
        return  2 * max_range * normalized_value - max_range
        
    def scale_to_normalized(self, actual_value, max_range):
        """Convert an actual value (within -max_range to +max_range) to a normalized value (0 to 1)"""
        max_range = float(max_range)
        return (actual_value + max_range) / (2 * max_range)
    
    def program_manual(self, front_panel_values): 
        """Allows for user control of the device via the BLACS_tab, 
        setting outputs to the values set in the BLACS_tab widgets. 
        Runs at the end of the shot."""
        
        print(f"front panel values: {front_panel_values}")
        
        for channel, value in front_panel_values.items():
            scaled_voltage = self.scale_to_normalized(float(value), float(self.device_voltage_range))
            sendStr = f"{self.device_serial} {channel} {scaled_voltage:.6f}\r"
            self.send_to_HV(sendStr)
            
        return front_panel_values

    def check_remote_values(self): # reads the current settings of the device, updating the BLACS_tab widgets 
        return

    def transition_to_buffered(self, device_name, h5_file, initial_values, fresh): 
        """transitions the device to buffered shot mode, 
        reading the shot h5 file and taking the saved instructions from 
        labscript_device.generate_code and sending the appropriate commands 
        to the hardware. 
        Runs at the start of each shot."""
        return 
        

    def transition_to_manual(self): 
        """transitions the device from buffered to manual mode to read/save measurements from hardware
        to the shot h5 file as results. 
        Runs at the end of the shot."""
        return  
    
    def send_to_HV(self, sendStr):
        logger.debug(f"Sent to HV_: {sendStr}")
        self.connection.write(sendStr.encode())
        
    def receive_from_HV(self):
        response = self.connection.readline().decode()
        logger.debug(f"Received from HV_: {response}")
        return(response)
        