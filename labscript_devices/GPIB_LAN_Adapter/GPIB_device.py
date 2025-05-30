from labscript import Device,set_passed_properties, LabscriptError
import GPIBLANAdapter
import numpy as np
import warnings


# This probably should be migrated to core
class GPIBDevice(Device):
    """Base class for all GPIB devices that are to be controlled by GPIB Adapters"""
    
    @set_passed_properties(property_names = {})
    def __init__(self, name,address, parent_device , **kwargs):
        
        if not isinstance(parent_device,GPIBLANAdapter):
            raise LabscriptError("Parent must be an instance of GPIBLANAdapter")
        
        self.name = name
        self.address = address

    def add_cmd(self,cmd):
        # TODO add command
        return 