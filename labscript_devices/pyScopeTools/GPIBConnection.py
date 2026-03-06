from __future__ import print_function
import visa
import time

"""
IMPORTANT: install pyVisa ! (e.g via pip install pyVisa). this requires that the NI-Visa driver is already installed
"""

def list_devices():
    """
    Helper function to list all valid addresses
    """
    rm = visa.ResourceManager()
    print(rm.list_resources())
    
class GPIBComm:
    def __init__(self, address, timeout=2000, eol='\r\n'):
        rm = visa.ResourceManager()
        try:
            self.inst = rm.open_resource(address)#, read_termination=eol)
        except:
            print("Cannot open connection.")
            
        if timeout:
            self.inst.timeout = timeout

        self.do_init_oszi()    
    
    def do_init_oszi(self):
        self.writeline("*CLS") #clear status

    def readline(self):
        return self.inst.read()
        
    def read(self, bytes=1):
        return self.readline_raw()
        #return self.connection.read(bytes)
        
    def readline_raw(self, min_bytes):
        return self.inst.read_raw()

    def write(self, message):
        self.writeline(message)
    
    def writeline(self, message):
        self.inst.write(message) #will append eol chars
        
    def query(self, msg, timeout=None):
        oldTimeout = self.inst.timeout
        if timeout:
            self.inst.timeout = timeout
        result = self.inst.query(msg)
        self.inst.timeout = oldTimeout
        return result    