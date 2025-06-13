from blacs.tab_base_classes import Worker, define_state
from blacs.device_base_class import DeviceTab
from qtutils.qt.QtWidgets import QPushButton, QSizePolicy as QSP, QHBoxLayout, QSpacerItem
from .logger_config import logger
from blacs.tab_base_classes import MODE_MANUAL

class HV_Tab(DeviceTab):
    def initialise_GUI(self):
        # Analog output properties dictionary
        connection_table = self.settings['connection_table']
        properties = connection_table.find_by_name(self.device_name).properties
        print(f" properties from connection table: {properties}")
        self.num_AO = properties['num_AO']
        self.base_unit = 'V'
        if self.num_AO > 0:
            self.base_min, self.base_max = properties['AO_range']
        else:
            self.base_min, self.base_max = None, None
        self.base_step = 10
        self.base_decimals = 3
        
        analog_properties = {}
        for i in range(1, self.num_AO + 1):
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
        _, ao_widgets, _ = self.auto_create_widgets()
        self.auto_place_widgets(("Analog outputs", ao_widgets))

        # Create buttons to send-to-device and check-remote-values
        self.send_button = self._create_button("Send to device", self.send_to_HV)
        self.check_button = self._create_button("Check remote values", self.check_remote_values)

        # Add centered layout to center the button
        center_layout = QHBoxLayout()
        center_layout.addStretch()
        center_layout.addWidget(self.send_button)
        center_layout.addWidget(self.check_button)
        center_layout.addStretch()

        # Add center layout on device layout
        self.get_tab_layout().addLayout(center_layout)

        self.supports_smart_programming(False)        
        self.supports_remote_value_check(False)
    
    def initialise_workers(self):
        # Get properties from connection table
        device = self.settings['connection_table'].find_by_name(self.device_name)

        if device is None:
            raise ValueError(f"Device '{self.device_name}' not found in the connection table.")
        
        # Look up at the connection table for device properties
        port = device.properties["port"]
        baud_rate = device.properties["baud_rate"]
        num_AO = device.properties['num_AO']
        worker_kwargs = {"name": self.device_name + '_main',
                         "port": port,
                         "baud_rate": baud_rate,
                         "num_AO": num_AO
                         }
        
        self.create_worker(
            'main_worker',
            'labscript_devices.HV_stahl.BLACS_workers.HV_Worker',
            worker_kwargs,
            )
        
        self.primary_worker = "main_worker"

    @define_state(MODE_MANUAL, True)
    def send_to_HV(self):
        """Queue a manual send-to-device operation from the GUI.

            This function is triggered from the BLACS tab (by pressing a button)
            and runs in the main thread. It queues the `send_to_HV()` function to be
            executed by the worker.

            Used to reprogram the device based on current front panel values.
            """
        try:
            yield (self.queue_work(self.primary_worker, 'send_to_HV', []))
        except Exception as e:
            logger.debug(f"Error by send work to worker(send_to_HV): \t {e}")

    def _create_button(self, text, on_click_callback):
        """Creates a styled QPushButton with consistent appearance and connects it to the given callback."""
        button = QPushButton(text)
        button.setSizePolicy(QSP.Fixed, QSP.Fixed)
        button.adjustSize()
        button.setStyleSheet("""
                QPushButton {
                    border: 1px solid #B8B8B8;
                    border-radius: 3px;
                    background-color: #F0F0F0;
                    padding: 4px 10px;
                    font-weight: light;
                }
                QPushButton:hover {
                    background-color: #E0E0E0;
                }
                QPushButton:pressed {
                    background-color: #D0D0D0;
                }
            """)
        button.clicked.connect(lambda: on_click_callback())
        logger.debug(f"Button {text} is created")
        return button
