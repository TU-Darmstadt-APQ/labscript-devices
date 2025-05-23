from blacs.tab_base_classes import Worker, define_state
from blacs.device_base_class import DeviceTab

class HV_Tab(DeviceTab):
    def initialise_GUI(self):
        # Analog output properties dictionary
        self.base_unit = 'V'
        self.base_min = -200 # TODO: Take from capabilities in models?
        self.base_max = 200
        self.base_step = 10
        self.base_decimals = 2
        self.num_AO = 4
        
        analog_properties = {}
        for i in range(self.num_AO):
            analog_properties['CH%d' % i] = {
                'base_unit': self.base_unit,
                'min': self.base_min,
                'max': self.base_max,
                'step':self.base_step,
                'decimals': self.base_decimals,
                }
        # Create and save AO objects
        self.create_analog_outputs(analog_properties)
        # Create widgets for AO objects
        dds_widgets, ao_widgets, do_widgets = self.auto_create_widgets()
        self.auto_place_widgets(("Analog outputs", ao_widgets))
        self.supports_smart_programming(False)        
        self.supports_remote_value_check(False)
    
    def initialise_workers(self):
        # Get properties from connection table
        device = self.settings['connection_table'].find_by_name(self.device_name)
        if device is None:
            raise ValueError(f"Device '{self.device_name}' not found in the connection table.")
        
        # Look up a the connection table for device properties
        port = device.properties["port"]
        baud_rate = device.properties["baud_rate"]
        worker_kwargs = {
            "name": self.device_name + '_main',
            "port": port,
            "baud_rate": baud_rate,
            }
        
        self.create_worker(
            'main_worker',
            'user_devices.HV_stahl.BLACS_workers.HV_Worker',
            worker_kwargs,
            )
        
        self.primary_worker = "main_worker"
        
