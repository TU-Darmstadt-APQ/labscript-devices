from labscript import IntermediateDevice, LabscriptError, set_passed_properties ,LabscriptError,set_passed_properties




class GPIBLANAdapter(IntermediateDevice):
    """
    A labscript_device TODO.
          - connection_table_properties (set once)
          - device_properties (set per shot)
    """

    @set_passed_properties(
        property_names = {
            'device_properties' : ["some_prop"]
            }
        )
    def __init__(self, 
                 name, 
                 address,
                 timeout = 2,
                 **kwargs):
        IntermediateDevice.__init__(self, name, parent_device =None, **kwargs) 

        self.BLACS_connection = address

        # --------------------------------- Class attributes
        self.name = name
        self.timeout = timeout
        self.some_prop = 2


    def generate_code(self, hdf5_file, *args):                  
        IntermediateDevice.generate_code(self, hdf5_file)

        # if self.configuration_number is not None:
        #     self.set_property('configuration_number', self.configuration_number, location='device_properties', overwrite=True)

