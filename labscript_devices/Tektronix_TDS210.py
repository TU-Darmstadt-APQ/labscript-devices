from labscript import LabscriptError, Device, set_passed_properties
from labscript_devices import BLACS_tab, runviewer_parser

import numpy as np
import labscript_utils.h5_lock
import h5py
import labscript_utils.properties

import logging
import logging.handlers


class Tektronix_TDS210(Device):

    def __init__(self, name, COM_port, **kwargs):
        self.BLACS_connection = COM_port
        self.ch1_enabled = False
        self.ch2_enabled = False
        self.oszi_settings = {}
        Device.__init__(self, name, None, self.BLACS_connection, **kwargs)

    def enable(self, ch1_enabled=True, ch2_enabled=True):
        """This method enables CH1 and/or CH2. This means that the corresponding waveforms are recorded in the shot"""
        self.ch1_enabled = ch1_enabled
        self.ch2_enabled = ch2_enabled

    def config_oszi(self, ch1_coupling="DC", ch1_position=0., ch1_scale=1., ch2_coupling="DC", ch2_position=0., ch2_scale=1., time_position=0., time_scale=0.01, trig_mode="NORMAL", trig_type="EDGE", trig_coupling="DC", trig_slope="RISE", trig_level=0.6):
        """
        Configure the scope.

        With this method you can configure the scope by setting up coupling, offsets, scalings and trigger settings.

        Parameters
        ----------
        ch1_coupling : string [DC, AC, GND]
            Set the coupling mode for CH1
        ch1_position : float
            Set the y-offset in V for CH1
        ch1_scale : float
            Set the scaling in V for CH1

        ch2_coupling : string [DC, AC, GND]
            Set the coupling mode for CH2
        ch2_position : float
            Set the y-offset in V for CH2
        ch2_scale : float
            Set the scaling in V for CH2

        time_position : float
            Set the x-offset in s for all channels (trigger point x-offset)
        time_scale : float
            Set the x-offset in s for all channels
        trig_mode : string [NORMAL, AUTO]
            Set the trigger mode
        trig_type : string [EDGE, VIDEO]
            Set the trigger type
        trig_coupling : string [AC, DC, NOISEREJ, HFREJ, NJREJ]
            Set the trigger coupling type
        trig_slope : string [RISE, FALL]
            Set the active edge for triggering
        trig_levek : float
            Set the trigger level in V
        """
        self.oszi_settings['ch1_coupling'] = ch1_coupling
        self.oszi_settings['ch1_position'] = ch1_position
        self.oszi_settings['ch1_scale'] = ch1_scale

        self.oszi_settings['ch2_coupling'] = ch2_coupling
        self.oszi_settings['ch2_position'] = ch2_position
        self.oszi_settings['ch2_scale'] = ch2_scale

        self.oszi_settings['time_position'] = time_position
        self.oszi_settings['time_scale'] = time_scale

        self.oszi_settings['trig_mode'] = trig_mode
        self.oszi_settings['trig_type'] = trig_type
        self.oszi_settings['trig_coupling'] = trig_coupling
        self.oszi_settings['trig_slope'] = trig_slope
        self.oszi_settings['trig_level'] = trig_level

    def generate_code(self, hdf5_file):
        """
        Store the configuration in the shot h5 file
        """
        group = self.init_device_group(hdf5_file)
        group.attrs['ch1_enabled'] = self.ch1_enabled
        group.attrs['ch2_enabled'] = self.ch2_enabled
        group.attrs['oszi_settings'] = str(self.oszi_settings)  # convert dict to str


import time

from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED
from blacs.device_base_class import DeviceTab

from qtutils import UiLoader
import os


@BLACS_tab
class Tektronix_TDS210Tab(DeviceTab):

    def initialise_GUI(self):
        """
        Initialises the BLACS device Tab
        """
        layout = self.get_tab_layout()
        ui_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'scope.ui')
        self.ui = UiLoader().load(ui_filepath)
        layout.addWidget(self.ui)

    def initialise_workers(self):
        """
        Initialise the BLACS worker to communicate with the device
        """
        worker_initialisation_kwargs = {'BLACS_connection': self.BLACS_connection}
        self.create_worker("main_worker", Tektronix_TDS210Worker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"

        self.statemachine_timeout_add(3000, self.update_oszi_display)  # delay the first communication (otherwise sometimes an error occures ?)

    @define_state(MODE_MANUAL, True)
    def update_oszi_display(self):
        """
        Request the worker for the firmware version and update the firmware label.
        """
        self.statemachine_timeout_remove(self.update_oszi_display)  # only update once

        oszi_id = yield(self.queue_work(self._primary_worker, 'get_oszi_id'))
        self.ui.lbl_firmware.setText(str(oszi_id))


class Tektronix_TDS210Worker(Worker):
    def init(self):
        global scope
        import labscript_devices.pyScopeTools.scope

        # instatiate the scope object
        self.scope = labscript_devices.pyScopeTools.scope.Scope(self.BLACS_connection, baudrate=9600, debug=False)

        self.h5file = None
        self.logger = logging.getLogger('BLACS')  # for debugging

        # self.settings_buffer = {} # This dict contains sub-dicts
        #self.settings_buffer['CH1']  = {}
        #self.settings_buffer['CH2']  = {}
        #self.settings_buffer['TIME'] = {}
        #self.settings_buffer['TRIG'] = {}
        self.settings_buffer = {'CH1': {}, 'CH2': {}, 'TIME': {}, 'TRIG': {}}  # should also do the trick, but faster

        self.enabled = True

        time.sleep(1)
        self.update_oszi_id()
        time.sleep(1)  # too much time?
        self.update_oszi_settings()
        time.sleep(1)  # too much time?

    def get_oszi_id(self):
        if 'ID' in self.settings_buffer:
            return self.settings_buffer['ID']
        else:
            return "Unkown Oszi ID"  # if the ID has not been requested

    def update_oszi_id(self):
        self.settings_buffer['ID'] = self.scope.get_oszi_ID().strip()  # crop whitespaces

    def update_oszi_settings(self):
        # the delays may be too long. but this method is only called once when BLACS/the worker starts up, so who cares ;-)
        channel1 = self.scope.get_channel_config(channel=1)
        time.sleep(1)
        channel2 = self.scope.get_channel_config(channel=2)
        self.settings_buffer['CH1'].update(channel1)  # update the internal settings dict
        self.settings_buffer['CH2'].update(channel2)  # update the internal settings dict
        time.sleep(1)
        horz = self.scope.get_time_config()
        self.settings_buffer['TIME'].update(horz)
        time.sleep(1)
        trig = self.scope.get_trigger_config()
        self.settings_buffer['TRIG'].update(trig)

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        """
        Load the scope settings from the shot h5 file and send them to the oszi, if necessary
        """
        self.h5file = h5file
        self.device_name = device_name
        settings = None
        ch1_enabled = False
        ch2_enabled = False
        with h5py.File(h5file, 'r') as hdf5_file:
            group = hdf5_file['devices/'][device_name]
            ch1_enabled = group.attrs['ch1_enabled']
            ch2_enabled = group.attrs['ch2_enabled']
            settings = eval(group.attrs['oszi_settings'])  # convert str back to dict

        self.enabed = ch1_enabled or ch2_enabled

        if self.enabled and settings:  # if there is no channel that should be recorded, the scope does not need to be configured at all
            # only send a command if the requested setting differs from the current scope setting
            ch1_coupling = None
            if self.settings_buffer['CH1']['COUPLING'] != settings['ch1_coupling']:
                ch1_coupling = settings['ch1_coupling']
                self.settings_buffer['CH1']['COUPLING'] = settings['ch1_coupling']
            ch1_position = None
            if self.settings_buffer['CH1']['POSITION'] != settings['ch1_position']:
                ch1_position = settings['ch1_position']
                self.settings_buffer['CH1']['POSITION'] = settings['ch1_position']
            ch1_scale = None
            if self.settings_buffer['CH1']['SCALE'] != settings['ch1_scale']:
                ch1_scale = settings['ch1_scale']
                self.settings_buffer['CH1']['SCALE'] = settings['ch1_scale']
            if self.scope.config_channel(channel=1, coupling=ch1_coupling, position=ch1_position, scale=ch1_scale):
                time.sleep(1)  # give the oszi time to execute the config, if a command is sent to the scope

            ch2_coupling = None
            if self.settings_buffer['CH2']['COUPLING'] != settings['ch2_coupling']:
                ch2_coupling = settings['ch2_coupling']
                self.settings_buffer['CH2']['COUPLING'] = settings['ch2_coupling']
            ch2_position = None
            if self.settings_buffer['CH2']['POSITION'] != settings['ch2_position']:
                ch2_position = settings['ch2_position']
                self.settings_buffer['CH2']['POSITION'] = settings['ch2_position']
            ch2_scale = None
            if self.settings_buffer['CH2']['SCALE'] != settings['ch2_scale']:
                ch2_scale = settings['ch2_scale']
                self.settings_buffer['CH2']['SCALE'] = settings['ch2_scale']
            if self.scope.config_channel(channel=2, coupling=ch2_coupling, position=ch2_position, scale=ch2_scale):
                time.sleep(1)  # give the oszi time to execute the config

            time_pos = None
            if self.settings_buffer['TIME']['POSITION'] != settings['time_position']:
                time_pos = settings['time_position']
                self.settings_buffer['TIME']['POSITION'] = settings['time_position']
            time_scale = None
            if self.settings_buffer['TIME']['SCALE'] != settings['time_scale']:
                time_scale = settings['time_scale']
                self.settings_buffer['TIME']['SCALE'] = settings['time_scale']
            if self.scope.config_time(position=time_pos, scale=time_scale):
                time.sleep(1)  # give the oszi time to execute the config

            trig_mode = None
            if self.settings_buffer['TRIG']['MODE'] != settings['trig_mode']:
                trig_mode = settings['trig_mode']
                self.settings_buffer['TRIG']['MODE'] = settings['trig_mode']
            trig_type = None
            if self.settings_buffer['TRIG']['TYPE'] != settings['trig_type']:
                trig_type = settings['trig_type']
                self.settings_buffer['TRIG']['TYPE'] = settings['trig_type']
            trig_coupling = None
            if self.settings_buffer['TRIG']['COUPLING'] != settings['trig_coupling']:
                trig_coupling = settings['trig_coupling']
                self.settings_buffer['TRIG']['COUPLING'] = settings['trig_coupling']
            trig_slope = None
            if self.settings_buffer['TRIG']['SLOPE'] != settings['trig_slope']:
                trig_slope = settings['trig_slope']
                self.settings_buffer['TRIG']['SLOPE'] = settings['trig_slope']
            trig_level = None
            if self.settings_buffer['TRIG']['LEVEL'] != settings['trig_level']:
                trig_level = settings['trig_level']
                self.settings_buffer['TRIG']['LEVEL'] = settings['trig_level']
            if self.scope.config_trigger(mode=trig_mode, typ=trig_type, coupling=trig_coupling, slope=trig_slope, level=trig_level):
                # time.sleep(1)
                pass
        if self.enabled:
            time.sleep(1)  # always sleep 1 sec so that the scope is ready
        return {}  # indicates final values of buffered run, we have no

    def transition_to_manual(self, abort=False):
        """
        This method reads the waveform from the scope and stores it in the shot h5 file
        """
        if not self.enabled:  # if no channel is enabled, there is no data to store in the h5 file. skip this
            return True

        annotation = ""
        with h5py.File(self.h5file, 'a') as hdf5_file:
            try:
                measurements = hdf5_file['/data/traces']
            except:
                # Group doesn't exist yet, create it:
                measurements = hdf5_file.create_group('/data/traces')
            attrs = hdf5_file['/devices/' + self.device_name].attrs
            requestString = ""
            if attrs['ch1_enabled']:
                requestString += "CH1"
            if attrs['ch2_enabled']:
                requestString += "CH2"

            if requestString:
                if len(requestString) > 3:  # request both channels
                    t, data1, data2 = self.scope.readScope(requestString)

                    data = np.vstack((t, data1, data2)).T
                else:
                    t, data1 = self.scope.readScope(requestString)
                    self.logger.info("TDS210_INFO: t len=" + str(t.shape))
                    self.logger.info("TDS210_INFO: y len=" + str(data1.shape))
                    diff = abs(t.shape[0] - data1.shape[0])
                    if diff > 0 and diff <= 3:
                        t = t[:-diff]  # crop t to size
                        annotation = "cropped"
                    data = np.vstack((t, data1)).T

                dataset = measurements.create_dataset("scope_waveform", data=data)  # create a dataset with the name 'scope_waveform' to store the waveform
                if annotation:
                    dataset.attrs["annotation"] = annotation

        return True  # indicates success

    def abort_buffered(self):
        return self.abort()

    def abort_transition_to_buffered(self):
        return self.abort()

    def abort(self):
        self.scope.unfreeze()
        return True  # indicates success

    def program_manual(self, values):
        return {}

    def shutdown(self):
        return
