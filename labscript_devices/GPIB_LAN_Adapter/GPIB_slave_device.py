from labscript import Device,set_passed_properties, LabscriptError
import numpy as np
import warnings


class GPIBSlaveDevice(Device):
    """ Base class for all GPIB devices that are to be controlled by GPIBLANAdapterDevice class """
    
    @set_passed_properties(property_names = {})
    def __init__(self, 
                 name, 
                 parent_device ,
                 connection,                        # Excample HP 6632A => connection =  17 
                 # called_parent_device =True,      # This is already True in Device # Delete Me Later
                 # limits ,                                 
                 **kwargs):
        

        Device.__init__(self,name,parent_device,connection, **kwargs)


    def cmd(self, cmd):
        """ Command the device, Override this methode in subclass """
        pass 


class HP6632A(GPIBSlaveDevice):
    ''' 
        Uses HP BASIC programming language

        Capabilities : 
            Voltage         :   0 - 20.475 (5e-3) V
            Current         :   0 - 5.1188 (125e-3) A
            Overvoltage     :   0 - 22V    (0.1) V
            
            !!! programmable are 2.375% Higher

        Language        : ASCII 
        Numerics        : 1.23E3
        Terminators     :  ;   LF   CR LF
    
    '''
    description = "HP6632A"

    # @set_passed_properties(property_names={}) # TODO
    def __init__(self,*args,**kwargs):
        Device.__init__(self,*args,**kwargs)

    def cmd(self):
        pass  # TODO

