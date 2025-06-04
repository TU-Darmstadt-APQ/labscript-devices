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
        
        self.name = name
        self.connection = connection


    def add_cmd_group(self,group):
        """ Override this methode in subclass"""
        pass
    
    def add_cmd(self,group, cmd):
        """ Override this methode in subclass """
        pass 


class HP6632A(GPIBSlaveDevice):
    ''' 
        Uses HP BASIC programming language

        Capabilities : 
            Voltage         :   0 - 20.475 (5e-3) V
            Current         :   0 - 5.1188 (125e-3) A
            Overvoltage     :   0 - 22V    (0.1) V
            
            !!! programmable are 2.375 Higher

        Language        : ASCII 
        Numerics        : 1.23E3
        Terminators     :  ;   LF   CR LF
    
    '''
    def __init__(self,*args,**kwargs):
        super.__init__(*args,**kwargs)

        self.type_device = "HP6632A"




# class PrologixGPIBEthernetDevice:
#     def __init__(self, address, *args, **kwargs):
#         self.address = address
#         self.gpib = PrologixGPIBEthernet(*args, **kwargs)

#     def connect(self):
#         self.gpib.connect()
#         self.gpib.select(self.address)

#     def close(self):
#         self.gpib.close()

#     def write(self, *args):
#         return self.gpib.write(*args)

#     def read(self, *args):
#         return self.gpib.read(*args)

#     def query(self, *args):
#         return self.gpib.query(*args)

#     def idn(self):
#         return self.query('*IDN?')

#     def reset(self):
#         self.write('*RST')
