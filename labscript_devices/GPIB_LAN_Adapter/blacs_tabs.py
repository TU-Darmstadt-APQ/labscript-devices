from blacs.device_base_class import DeviceTab
from labscript import LabscriptError    
from blacs.tab_base_classes import define_state,Worker
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED  
from blacs.device_base_class import DeviceTab

import os
import sys
from PyQt5.QtWidgets import * 
from PyQt5.QtCore import QSize
from PyQt5 import uic


class GPIBLANAdapterTab(DeviceTab): 
    '''The device class handles the creation + interaction with the GUI ~ QueueManager'''

    def initialise_workers(self):
        # Here we can change the initialization properties in the connection table
        worker_initialisation_kwargs = self.connection_table.find_by_name(self.device_name).properties

        # Adding porperties as follows allows the blacs worker to access them
        # This comes in handy for the device initialization
        worker_initialisation_kwargs['address'] = self.BLACS_connection   

        # Create the device worker
        self.create_worker(
            'main_worker',
            'labscript_devices.GPIB_LAN_Adapter.blacs_workers.GPIBLANAdapterWorker',
            worker_initialisation_kwargs,
        )
        self.primary_worker = 'main_worker'


    def initialise_GUI(self):
        return







