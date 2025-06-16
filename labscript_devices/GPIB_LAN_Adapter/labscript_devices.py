from labscript import IntermediateDevice, LabscriptError, set_passed_properties ,LabscriptError,set_passed_properties
from labscript_devices.GPIB_LAN_Adapter.GPIB_slave_device import GPIBSlaveDevice

from numpy import empty


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
        # --------------------------------- class attributes
        self.name = name
        self.timeout = timeout
        IntermediateDevice.__init__(self, name, parent_device =None, **kwargs) 

    # Override methode add_device from Device -> IntermediatedDevice  (called automatically upon instantiating)
    def add_device(self,device):
        if not isinstance(device,GPIBSlaveDevice):
            raise LabscriptError("Parent must be an instance of GPIBLANAdapter")
        
        # Here : check device.connection ? 
        IntermediateDevice.add_device(self, device)


    def _get_info_child(self, device):
        info_dtypes = [
            ('connection', 'a256'),
            ('name', 'a256'),
        ]

        info_table = empty(1, dtype=info_dtypes)
        info_table[0] = (device.connection, device.name)
        return info_table
        

        
    def generate_code(self, hdf5_file, *args):                  
        IntermediateDevice.generate_code(self, hdf5_file)


        # if self.configuration_number is not None:
        #     self.set_property('configuration_number', self.configuration_number, location='device_properties', overwrite=True)

        # Logic copied from NI_DAQmx(Device) -> But I Think makes no sense here
        
        # slaves = {}
        # for device in self.child_devices:
        #     if isinstance(device,GPIBSlaveDevice):
        #         slaves[device.connection] = device
        #     else:
        #         raise TypeError(device)
            
        slaves = self.child_devices
        grp = self.init_device_group(hdf5_file)
        if len(slaves) == 0:
            return 
        if len(slaves) == 1:
            slave = slaves[0]
            if not isinstance(slave,GPIBSlaveDevice):
                raise TypeError(slave)
            grp.create_dataset(slave.name ,data = self._get_info_child(slaves[0])) 
        else:
            raise LabscriptError("Only connection to a single device is currently supported")

        





