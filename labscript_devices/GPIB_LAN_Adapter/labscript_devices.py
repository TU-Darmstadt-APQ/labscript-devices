from labscript import IntermediateDevice, LabscriptError, set_passed_properties ,LabscriptError,set_passed_properties
from labscript_devices.GPIB_LAN_Adapter.GPIB_slave_device import GPIBSlaveDevice


class GPIBLANAdapter(IntermediateDevice):
    """
    A labscript_device to manage GPIB devices via a prologix_alike Ethernet-GPIB adapter.
          - connection_table_properties (set once)
          - device_properties (set per shot)
    """

    allowed_children = [GPIBSlaveDevice]
    description = 'Prologix_alike_adapter'

    @set_passed_properties( property_names = {} )
    def __init__(self, 
                 name, 
                 ip_address,
                 timeout = 2,
                 **kwargs):
        
        self.BLACS_connection = ip_address
        # --------------------------------- Class attributes
        self.name = name
        self.timeout = timeout
        IntermediateDevice.__init__(self, name, parent_device =None, **kwargs) 

    # Override methode add_device from Device -> IntermediatedDevice  (called automatically upon instantiating)
    def add_device(self,device):
        if not isinstance(device,GPIBSlaveDevice):
            raise LabscriptError("Parent must be an instance of GPIBLANAdapter")
        IntermediateDevice.add_device(self, device)


    def get_infos_from_kids(self):
        return
        
    def generate_code(self, hdf5_file, *args):                  
        IntermediateDevice.generate_code(self, hdf5_file)

        # if self.configuration_number is not None:
        #     self.set_property('configuration_number', self.configuration_number, location='device_properties', overwrite=True)

