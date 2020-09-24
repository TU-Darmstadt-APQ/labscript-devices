#####################################################################
#                                                                   #
# /labscript_devices/AndorCamera.py                                      #
#                                                                   #
# Copyright 2013, Monash University                                 #
#                                                                   #
# This file is part of labscript_devices, in the labscript suite    #
# (see http://labscriptsuite.org), and is licensed under the        #
# Simplified BSD License. See the license.txt file in the root of   #
# the project for the full license.                                 #
#                                                                   #
#####################################################################
# try:
#     from labscript_utils import check_version
# except ImportError:
#     raise ImportError('Require labscript_utils > 2.1.0')

# check_version('labscript', '2.0.1', '3')

from labscript_devices.Camera import *
from labscript import set_passed_properties


class AndorCamera(Camera):
    description = 'Andor Camera'

    # To be set as instantiation arguments:
    trigger_edge_type = None
    minimum_recovery_time = None

    @set_passed_properties(
        property_names={"device_properties": ["gain", "save", "vsamplitude", "vsspeed", "hsspeed", "triggerMode",
                                              "shutterMode", "acquisitionMode", "accCycleTime", "accNum", "kinCycleTime",
                                              "kinNum", "frameTransfer"]}
    )
    def __init__(self, name, parent_device, connection, gain=0, save=1, vsspeed=3.4, hsspeed=3.0,
                 vsamplitude=0, triggerMode=1, shutterMode=1, acquisitionMode=5, accCycleTime=1.0,
                 accNum=1, kinCycleTime=0, kinNum=2000, frameTransfer=0, **kwargs):
        Camera.__init__(self, name, parent_device, connection, **kwargs)
        self.save = save
        self.gain = gain
        self.vsspeed = vsspeed
        self.hsspeed = hsspeed
        self.vsamplitude = vsamplitude

    def expose(self, name, t, frametype):
        return Camera.expose(self, name, t, frametype)


@BLACS_tab
class AndorCameraTab(CameraTab):
    pass


class AndorCameraWorker(CameraWorker):
    pass
