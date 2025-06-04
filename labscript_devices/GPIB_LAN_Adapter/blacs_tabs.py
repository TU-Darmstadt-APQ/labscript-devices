from blacs.device_base_class import DeviceTab
from labscript import LabscriptError    
from blacs.tab_base_classes import define_state,Worker
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED  
from blacs.device_base_class import DeviceTab

import os
import sys
from PyQt5.QtWidgets import QWidget
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
        # The osci widget
        self.adap_widget = GPIBLANGUI()
        self.get_tab_layout().addWidget(self.adap_widget)

        # --- Init
        self.init_slave_device()

    @define_state(MODE_MANUAL,True,True)
    def init_slave_device(self, widget=None ):
        device_name = yield(self.queue_work(self._primary_worker,'get_device_name'))
        address_gpib =  yield(self.queue_work(self._primary_worker,'get_address_gpib'))
        self.adap_widget.write_device_name(device_name)
        self.adap_widget.write_address_gpib(address_gpib)


class GPIBLANGUI(QWidget):
    """ The Adapter Widget """
    def __init__(self, parent=None):
        super().__init__(parent) 

        tabs_name = 'adapter_ui.ui'
        tabs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),tabs_name)
        uic.loadUi(tabs_path,self) 


    def write_device_name(self,device_name):
        self.device_name_change.setText(device_name)

    def write_address_gpib(self,address_gpib):
        self.gpib_addr_label_change.setText(address_gpib)







