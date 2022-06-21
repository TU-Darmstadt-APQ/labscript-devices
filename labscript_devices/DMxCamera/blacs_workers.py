#####################################################################
#                                                                   #
# /labscript_devices/DMxCamera/blacs_workers.py                     #
#                                                                   #
# Copyright 2019, Monash University and contributors                #
#                                                                   #
# This file is part of labscript_devices, in the labscript suite    #
# (see http://labscriptsuite.org), and is licensed under the        #
# Simplified BSD License. See the license.txt file in the root of   #
# the project for the full license.                                 #
#                                                                   #
#####################################################################
# This DMx implementation is based on the FlyCapture implementation from David Meyer.

# This code is still in development stage which is why here are some remarks:
# There is no function from the API to aks for new images. When using GetImage()
# the Buffer is just continously read out even if there are no ´new images. This
# is why for buffered acquisition the following method was used: The value of
# the pointer is read out as this value changes if a new image is captured due
# to a trigger send to the camera.
# In addition: To succesfully run the DMx camera in buffered mode, a dummy
# trigger has to be implemented at the beginning of the experiment as the first
# trigger is not recognized for image acquisition.


import ctypes as C
import numpy as np
import cv2
from labscript_utils import dedent
from enum import IntEnum
import sys
import os
import threading
import labscript_utils.h5_lock
import h5py
import time

from labscript_devices.IMAQdxCamera.blacs_workers import IMAQdxCameraWorker
from labscript_utils.shared_drive import path_to_local

# Don't import API yet so as not to throw an error, allow worker to run as a dummy
# device, or for subclasses to import this module to inherit classes without requiring API
IC = None


class DMx_Camera(object):
    def __init__(self, serial_number):
        """Initialize DMx API camera.
        Serial number should be of int type."""

        global IC
        import sys
        sys.path.append('C:\\labscript_suite\\labscript_devices\\DMxCamera\\')
        import tisgrabber as IC

        # Check for serial number as only one camera is implemented:
        checked_serial_numbers = [32800009]
        if serial_number not in checked_serial_numbers:
            msg = """This specific camera is not supported yet. Íf it is a model
            DMx 21BU04 camera please add its serial number to the blacs_worker_code."""
            raise RuntimeError(dedent(msg))

        # Create camera object:
        self.camera = IC.TIS_CAM()
        self.camera.open('DMx 21BU04 ' + str(serial_number))  # put it into try excdept block?

        if self.camera.IsDevValid() != 1:
            raise RuntimeError('Device does not exist')

        self._abort_acquisition = False

    def set_attributes(self, attr_dict):
        """Sets all attribues in attr_dict.
        TIS controls the camera using three different interfaces: Range, Switch and AbsoluteValue. Each Property is therefore defined by 4 ID's: "Item", "Element", "Interface" and "Value"
        Camera attributes need to be given as 'Item::Element::Interface': Value in the attributes dictionary.
        """

        for name, val in attr_dict.items():
            self.set_attribute(name, val)

    def set_attribute(self, name, value):
        """Set the values of the attribute of the given name using the provided
        dictionary values. Typical structure is:
        name: value where name = 'Item::Element::Interface'; e.g.: 'Exposure::Value::Range'
        """
        try:
            prop, element, interface = name.split('::')
        except ValueError:
            msg = f"""camera attribute {name} is not in the form 'property::element::interface'"""
            raise ValueError(dedent(msg))
        if interface == 'AbsoluteValue':
            self.camera.SetPropertyAbsoluteValue(prop, element, value)
        elif interface == 'Range':
            self.camera.SetPropertyValue(prop, element, value)
        elif interface == 'Switch':
            self.camera.SetPropertySwitch(prop, element, value)
        else:
            msg = f"""{name} has invalid interface {interface}. Interface must be 'Range', 'Switch' or 'AbsoluteValue'"""
            raise ValueError(dedent(msg))

    def get_attributes(self):
        """Return a nested dict of all readable attributes.
        The readable attributes have to be saved in prop_list
        as there is no function to read out the attributes
        from the corresponding camera.
        """
        prop_list = (
            'Brightness::Value::Range',
            'Gamma::Value::Range',
            'Gain::Value::Range',
            'Gain::Auto::Switch',
            'Exposure::Value::Range',
            'Exposure::Value::AbsoluteValue',
            'Exposure::Auto::Switch',
            'Exposure::Auto Reference::Range',
            'Exposure::Auto Max Value::Range',
            'Exposure::Auto Max Value::AbsoluteValue',
            'Trigger::Enable::Switch',
            'Trigger::Polarity::Switch'
        )
        props = {}
        for name in prop_list:
            props[name] = self.get_attribute(name)

        return props

    def get_attribute(self, name):
        """Return current values dictionary of attribute of the given name"""
        try:
            prop, element, interface = name.split('::')
        except ValueError:
            msg = f"""camera attribute {name} is not in the form 'property::element::interface'"""
            raise ValueError(dedent(msg))
        if interface == 'AbsoluteValue':
            value, error = self.camera.GetPropertyAbsoluteValue(prop, element)
        elif interface == 'Range':
            value, error = self.camera.GetPropertyValue(prop, element)
        elif interface == 'Switch':
            value, error = self.camera.GetPropertySwitch(prop, element)
        else:
            msg = f"""{name} has invalid interface {interface}. Interface must be 'Range', 'Switch' or 'AbsoluteValue'"""
            raise ValueError(dedent(msg))
        if error == 0:
            msg = f"""An error occuered when getiing the value of {name}."""
            raise Exception(dedent(msg))
        else:
            return value

    def snap(self):
        """Acquire a single image and return it"""

        self.configure_acquisition(continuous=True)
        img = self.grab()
        self.stop_acquisition()
        return img

    def configure_acquisition(self, continuous=True, bufferCount=1):
        """Configure continuous mode. Buffer count not needed for DMx camera.
        """
        if continuous:
            self.camera.SetContinuousMode(1)  # Images must be copied into memory by using snapImage(); Mode0 is necessary in triggered mode
        else:
            self.camera.SetContinuousMode(0)

        self.width = self.camera.get_video_format_width()
        self.height = self.camera.get_video_format_height()

        self.camera.StartLive(0)
        init_ptr = self.camera.GetImagePtr()
        print(f"Initial Ptr: {init_ptr}")

    def grab(self):
        """Grab and return single image during pre-configured acquisition."""

        try:
            error = self.camera.SnapImage()
        except error == 0:
            raise RuntimeError('Failed to snap')
        img = self.simplify_image_data(self.camera.GetImage())
        return img

    def grab_multiple(self, n_images, images, dt):
        """Grab n_images into images array during buffered acquisition. A
        dummy trigger is necesary at the beginning of the experiment for
        proper function. Furthermore the sleep time implemented is necessary
         to circumvent blacs to hang up."""
        print(f"Attempting to grab {n_images} images.")
        init_ptr = self.camera.GetImagePtr()
        print(f"Initial Ptr: {init_ptr}")
        print(dt)
        for i in range(n_images):
            while True:
                if self._abort_acquisition:
                    print("Abort during acquisition.")
                    self._abort_acquisition = False
                    return
                ptr = self.camera.GetImagePtr()
                if ptr != init_ptr:
                    # print("pointers not equal")
                    img = self.simplify_image_data(self.camera.GetImage())
                    images.append(img)
                    print(f"Current Ptr: {ptr}")
                    init_ptr = ptr
                    # self._send_image_to_parent(img)
                    print(f"Got image {i+1} of {n_images}.")
                    time.sleep(dt / 10)
                    break
                else:
                    # print("pointers equal")
                    time.sleep(dt / 10)
                    continue
        print(f"Got {len(images)} of {n_images} images.")

    def simplify_image_data(self, img):
        """DMx camera returns an image with 3 same layers.
        Reduce image to one layer."""
        return np.ascontiguousarray(img[:, :, 0])

    def stop_acquisition(self):
        self.camera.StopLive()

    def abort_acquisition(self):
        self._abort_acquisition = True

    def close(self):
        return
        # self.camera.disconnect()


class DMxCameraWorker(IMAQdxCameraWorker):
    """DMx API Camera Worker.

    Inherits from IMAQdxCameraWorker. Overloads get_attributes_as_dict
    to use DMxCamera.get_attributes() method."""
    interface_class = DMx_Camera

    def get_attributes_as_dict(self, visibility_level):
        """Return a dict of the attributes of the camera; visibility level nor used with DMx camera"""
        return self.camera.get_attributes()

    def get_attributes_as_text(self, visibility_level):
        """Return a string representation of the attributes of the camera; visibility level nor used with DMx camera"""
        attrs = self.get_attributes_as_dict(visibility_level)
        # Format it nicely:
        lines = [f'    {repr(key)}: {repr(value)},' for key, value in attrs.items()]
        dict_repr = '\n'.join(['{'] + lines + ['}'])
        return self.device_name + '_camera_attributes = ' + dict_repr

    # Copied from IMAQdx Camera:
    def transition_to_buffered(self, device_name, h5_filepath, initial_values, fresh):
        if getattr(self, 'is_remote', False):
            h5_filepath = path_to_local(h5_filepath)
        if self.continuous_thread is not None:
            # Pause continuous acquisition during transition_to_buffered:
            self.stop_continuous(pause=True)
        with h5py.File(h5_filepath, 'r') as f:
            group = f['devices'][self.device_name]
            if not 'EXPOSURES' in group:
                return {}
            self.h5_filepath = h5_filepath
            self.exposures = group['EXPOSURES'][:]
            self.n_images = len(self.exposures)

            # Get the camera_attributes from the device_properties
            properties = labscript_utils.properties.get(
                f, self.device_name, 'device_properties'
            )
            camera_attributes = properties['camera_attributes']
            self.stop_acquisition_timeout = properties['stop_acquisition_timeout']
            self.exception_on_failed_shot = properties['exception_on_failed_shot']
            self.exposure = camera_attributes['Exposure::Value::AbsoluteValue']
            saved_attr_level = properties['saved_attribute_visibility_level']
        # Only reprogram attributes that differ from those last programmed in, or all of
        # them if a fresh reprogramming was requested:
        if fresh:
            self.smart_cache = {}
        self.set_attributes_smart(camera_attributes)
        # Get the camera attributes, so that we can save them to the H5 file:
        if saved_attr_level is not None:
            self.attributes_to_save = self.get_attributes_as_dict(saved_attr_level)
        else:
            self.attributes_to_save = None
        print(f"Configuring camera for {self.n_images} images.")
        self.camera.configure_acquisition(continuous=False, bufferCount=self.n_images)
        self.images = []
        self.acquisition_thread = threading.Thread(
            target=self.camera.grab_multiple,
            # add exposure time variable as argument
            args=(self.n_images, self.images, self.exposure),
            daemon=True,
        )
        print(f"Configuring camera for {self.n_images} images.")
        self.acquisition_thread.start()
        return {}
