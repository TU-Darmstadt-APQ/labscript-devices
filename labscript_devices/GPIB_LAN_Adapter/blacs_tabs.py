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
        self.connection_table = self.settings['connection_table']
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
        self.adap_widget = GPIBLANGUI()
        self.get_tab_layout().addWidget(self.adap_widget)

        # --- Init
        self.init_slave_device()


    @define_state(MODE_MANUAL,True,True)
    def init_slave_device(self, widget=None ):
        device_name = yield(self.queue_work(self._primary_worker,'get_device_name'))
        address_gpib =  yield(self.queue_work(self._primary_worker,'get_address_gpib'))
        self.adap_widget.show_device_name(device_name)
        self.adap_widget.show_address_gpib(address_gpib)
        self.adap_widget.show_slave_response(device_name)


    @define_state(MODE_MANUAL,True,True)
    def send_cmd(self,cmd= None, widget=None):
        response = yield(self.queue_work(self._primary_worker,'send_cmd',cmd))
        return response

    def get_child_from_connection_table(self, parent_device_name, port):
        return DeviceTab.get_child_from_connection_table(self, parent_device_name, port)



class GPIBLANGUI(QWidget):
    """ The Adapter Widget """
    def __init__(self, parent=None):
        super().__init__(parent) 

        tabs_name = 'adapter_ui.ui'
        tabs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),tabs_name)
        uic.loadUi(tabs_path,self) 

        self.send_cmd_btn.clicked.connect(self.send_clicked)

    def keyPressEvent(self, event):
        if event.key() == 16777220:  
            self.show_slave_response("Enter Key Pressed") 
            
    def show_device_name(self,device_name):
        self.device_name_change.setText(device_name)

    def show_address_gpib(self,address_gpib):
        self.gpib_addr_label_change.setText(address_gpib)

    def show_slave_response(self,response):
        self.response_output.setText(response)

    def send_clicked(self):
        self.response_output.setText("clicked")




    







