

from blacs.device_base_class import DeviceTab
from labscript import LabscriptError 
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED 
  
from blacs.tab_base_classes import define_state,Worker


import os
import sys

from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QMessageBox

from agilent_4422B_device import AgilentE4422BDevice, agilent_e4422b_specs
from basic_device import RFGeneratorSpecs, RFGeneratorStats, Stats, UnitFreq, UnitPower


class HPE4422BTab(DeviceTab):

    # -------------------------------------------------- Helpers 
    # NOTE Just in case I change my mind  
    # def _get_analog_properties(self):
    #     specs = agilent_e4422b_specs
    #     analog_properties = {}

    #     analog_properties["rf/frequency"]   = { "base_unit": specs.frequency.unit,
    #                                             "min": specs.frequency.minimum,
    #                                             "max": specs.frequency.maximum,
    #                                             "step": 1.0,
    #                                             "decimals": 2,
    #                                         }
    #     analog_properties["rf/power"]       = { "base_unit": specs.power_dbm.unit,
    #                                             "min": specs.power_dbm.minimum,
    #                                             "max": specs.power_dbm.maximum,
    #                                             "step": 0.1,
    #                                             "decimals": 2,
    #                                         }
    #     return analog_properties
    

    def _connect_ui_signals(self, gen_widget):
        gen_widget.buttonApply.clicked.connect(self.apply_to_device)
        gen_widget.buttonReadDevice.clicked.connect(self.read_stats_from_device)
        gen_widget.buttonRfOn.clicked.connect(lambda: self.set_output_rf(True))
        gen_widget.buttonRfOff.clicked.connect(lambda: self.set_output_rf(False))

    def initialise_workers(self):
        worker_initialisation_kwargs = {'GPIB_address': self.GPIB_address}
        self.create_worker("main_worker", 'labscript_devices.HP_E4422B_RF_GEN.blacs_workers.AgilentE4422BWorker' , worker_initialisation_kwargs)
        self.primary_worker = "main_worker"


    def initialise_GUI(self):

        # --- Connection table properties
        connection_table        = self.settings['connection_table']
        connection_table_entry  = connection_table.find_by_name(self.device_name) 
        self.GPIB_address       = connection_table_entry.BLACS_connection

        # analog_properties = self._get_analog_properties() # NOTE  just in case 

        # -------------------------------------------------- UI
        # --- Rf Gen Widget
        self.gen_widget = AgilentE4422BWidget()
        self.get_tab_layout().addWidget(self.gen_widget)

        # --- Connecting The Wiedget
        self._connect_ui_signals(self.gen_widget)



    @define_state(MODE_MANUAL, True)
    def transition_to_buffered(self, h5_filepath, notify_queue):
        # for remote worker to find correct find path:
        if getattr(self, 'is_remote', False):
            h5_filepath = path_to_local(h5_filepath)
        DeviceTab.transition_to_buffered(self, h5_filepath, notify_queue)

    @define_state(MODE_BUFFERED, False)
    def transition_to_manual(self, notify_queue, program=False):
        DeviceTab.transition_to_manual(self, notify_queue, program)


    # -------------------------------------------------- UI METHODES 
    @define_state(MODE_MANUAL, True,True)
    def apply_to_device(self, widget=None):
        stats_to_apply : RFGeneratorStats = self.gen_widget.get_stats_to_apply()
        yield(self.queue_work(self._primary_worker,'apply_to_device',stats_to_apply)) # TODO Implement in Worker

    @define_state(MODE_MANUAL, True,True)
    def read_stats_from_device(self, widget=None):
        stats : RFGeneratorStats = yield(self.queue_work(self._primary_worker,'read_stats_from_device')) # TODO Implement in Worker
        self.gen_widget.refresh_stats(stats)

    @define_state(MODE_MANUAL,  True,True)
    def set_output_rf(self, state: bool, widget=None):
        yield(self.queue_work(self._primary_worker,'set_output_rf', state))     # TODO in the worker 




######################################################################################################
#                                           GUI                                                      #
######################################################################################################


class AgilentE4422BWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        ui_path ="agilent_e4422b_minimal.ui"
        uic.loadUi(ui_path, self)

        self._connect_signals()
        self._mark_values_not_applied()

    # --------------------------------------------------
    # --- UI State
    def _mark_values_not_applied(self):
        self.labelApplyInfo.setText("Edited values are not sent to the device until Apply is pressed.")

    def _mark_values_applied(self):
        self.labelApplyInfo.setText( "Current edited values were applied to the device.")

    def _show_error(self, title, error):
        QMessageBox.critical(self, title, str(error))
        raise LabscriptError(error)
    
    # --- Signals
    def _connect_signals(self):
        self.spinFrequencyMHz.valueChanged.connect(self._mark_values_not_applied)
        self.spinPowerDbm.valueChanged.connect(self._mark_values_not_applied)
        self.buttonApply.clicked.connet(self._mark_values_applied)

    # --------------------------------------------------
    # --- Device Actions
    def get_stats_to_apply(self) -> RFGeneratorStats :
        freq_mhz = self.spinFrequencyMHz.value()
        power_dbm = self.spinPowerDbm.value()
        return RFGeneratorStats(freq_mhz=freq_mhz , power_dbm=power_dbm)

    def refresh_stats(self, stats : RFGeneratorStats ):
        self.labelFrequencyReadback.setText(f"{stats.freq_mhz:.3f} MHz")
        self.labelPowerReadback.setText(f"{stats.power_dbm:.2f} dBm")
        self.labelRfReadback.setText("ON" if stats.rf_on else "OFF")

