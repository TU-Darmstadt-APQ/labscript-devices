
import numpy as np
import h5py
from zprocess import rich_print
from blacs.tab_base_classes import Worker
from labscript_utils import properties



class GPIBLANAdapterWorker(Worker):
    """
    Defines the software control interface to the hardware. 
    The BLACS_tab spawns a process that uses this class to communicate with the hardware.
    """
    def init(self):
        # ----------------------------------------- Initialize
        global GPIBLANAdapterDevice
        from .GPIBLANAdapter import GPIBLANAdapterDevice

        self.adap = GPIBLANAdapterDevice(
            address= self.address,
            timeout= 1
            )

        # ----------------------------------------- Configurations attributes


        # ----------------------------------------- Buffered/Manuel flags
        

    def transition_to_buffered( self, device_name, h5file , front_panel_values, refresh): 
        rich_print(f"====== Begin transition to Buffered: ======", color='#66D9EF')    
        
        self.h5file = h5file                                                    
        self.device_name = device_name

        # with h5py.File(self.h5file, 'r+') as f:
            # ----------------------------------------- Get device properties
            # self.triggered = properties.get(f, device_name, 'device_properties')["some_prop"]


        rich_print(f"====== End transition to Buffered: ======", color='#66D9EF') 
        return {}

    def transition_to_manual(self, abort = False):
        rich_print(f"====== Begin transition to manual: ======", color='#A6E22E')
      
        # with h5py.File(self.h5file, 'r+') as hdf_file:          # r+ : Read/write, file must already exist 
        #     grp = hdf_file.create_group('/foo/bar')

        rich_print(f"====== End transition to manual: ======", color='#A6E22E')
        return True

    # ----------------------------------------- Aborting
    def abort_transition_to_buffered(self):
        """Special abort shot configuration code belongs here.
        """
        return self.transition_to_manual(True)
        
    def abort_buffered(self):
        """Special abort shot code belongs here.
        """
        return self.transition_to_manual(True)

    # ----------------------------------------- Override for remote 
    def program_manual(self,front_panel_values):
        """Over-ride this method if remote programming is supported.
        
        Returns:
            :obj:`check_remote_values()`
        """
        return self.check_remote_values()

    def check_remote_values(self):
        # over-ride this method if remote value check is supported
        return {}
    
    # ------------------------------------------ Blacs Tabs functions
    def shutdown(self):
        rich_print(f"====== transition to manual: ======", color= '#AE81FF')
        return 
    
    # -----
    def get_address_gpib(self):
        return str(self.adap.get_address_gpib()).strip()
    

    def get_device_name(self):
        return "Dummy Device"

    
