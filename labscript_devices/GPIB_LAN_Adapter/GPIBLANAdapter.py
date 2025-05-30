import pyvisa
import numpy as np
from labscript.labscript import LabscriptError


class GPIBLANAdapterDevice: 
    def __init__(self,
                 address,
                 verbose = True
                 ):
        
        self.verbose = verbose
        self.address = address

        # --------------------------------- Connecting to device 

        
        # --------------------------------- Initialize device


    #######################################################################################
    #                             Saving and Recalling                                    #
    ####################################################################################### 
    def foo(self):
        return
