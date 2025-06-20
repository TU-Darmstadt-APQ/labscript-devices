from blacs.tab_base_classes import Worker
from labscript.labscript import LabscriptError
from socket import socket,error , AF_INET, SOCK_STREAM,SHUT_RDWR


class GPIBWorker(Worker):

    PORT = 1234

    def init(self):
        global visa
        import pyvisa as visa
        global h5py
        import labscript_utils.h5_lock
        import h5py

        self.rm = None
        self.GPIB_connection = None
    
        self.init_GPIB()
        
    def init_GPIB(self):
        # Case 1 : GPIB 
        if self.GPIB_address.startswith("GPIB"):
            try:
                self.GPIB_address = self.GPIB_address +"::INSTR"
                self.rm = visa.ResourceManager()
                self.GPIB_connection = self.rm.open_resource(self.GPIB_address)
                self.GPIB_connection.lock(requested_key=self.GPIB_address)      
                self.GPIB_connection.lock_excl()
            except Exception as e:
                raise LabscriptError(f"Error with pyvisa connection : {e}")
        
        # Case 2 : Adapter
        elif self.GPIB_address.startswith("ADAP"):
            try:
                _ , ip_add , gpib_add  =  self.GPIB_address.split("::",2)
                self.GPIB_connection = AdapterConnection(self.device_name , ip_add , gpib_add)
            except Exception as e:
                raise LabscriptError(f"Error with Adapter connection : {e}")
        else:
            raise LabscriptError(f"Failed connection to {self.device_name} ")
            

    def shutdown(self):
        if self.GPIB_connection is not None:
            try:
                self.GPIB_connection.unlock()  
                self.GPIB_connection.shutdown()
                self.GPIB_connection.close()
                self.GPIB_connection = None
            except Exception as e:
                raise LabscriptError(f"Failed shutting down : {e}")

            if self.rm is not None:
                self.rm.close()  
                self.rm = None



    def abort_transition_to_buffered(self):
        return self.transition_to_manual()

    def abort_buffered(self):
        return self.transition_to_manual()

    def transition_to_manual(self):
        return True  
    

    
    

class AdapterConnection:
    '''Connection class to a Kofotronic (Prologix alike) adapter, that imitates a GPIB connection from pyvisa'''

    PORT = 1234

    def __init__(self, device_name, ip_ad, gpib_ad,timeout=1):
        self.device_name = device_name
        self.ip_ad      = ip_ad
        self.gpib_ad    = gpib_ad 
        # --------------------------------- Connecting and Initialize Adapter 
        self._socket  = socket(AF_INET,SOCK_STREAM)
        self.set_timeout(timeout)              # Socket communication timeout # TODO not flexible
        self._init_adapter()


    def set_timeout(self, value):   # https://prologix.biz/downloads/PrologixGpibEthernetManual.pdf#page=13
        'Timeout must be >= 1e-3 (1ms) and <= 3 (3s)'
        if value < 1e-3 or value > 3:
            raise LabscriptError('Timeout must be >= 1e-3 (1ms) and <= 3 (3s)')
        self._socket.settimeout(value)

    # --- Connect Adapter
    def _connect_to_adapter(self):
        try :
            self._socket.connect((self.ip_ad, self.PORT))
            print(f"✅ Adpater : {self.ip_ad}")
        except Exception as e:
            raise LabscriptError(f"Connection to Adapter not successful :{e}")
        
    # -- Connect to Instrument
    def _connect_instr(self):
        try:
            self._send(f'++addr {self.gpib_ad}')
            print(f"✅ {self.device_name}")
        except Exception as e:
            raise LabscriptError(f"Connection to Instrument not successful :{e}")
        
    # --- Send and Receive socket
    def _send(self, cmd):
        try:
            encoded_value = ('%s\n' % cmd).encode('ascii')
            self._socket.sendall(encoded_value)
        except Exception as e:
            raise LabscriptError(f"Failed to send command :{e}")

    def _recv(self, byte_num=1024):
        value = self._socket.recv(byte_num)
        return value.decode('ascii')
    
    # --- Setup
    def _setup_as_controller(self):
        self._send( "++mode 1")                  # 0 Device Mode         # 1 Controller
        self._connect_instr()
        self._send( "++auto 0")                  # 0 Instrument LISTEN   # 1 Instrument TALK.
        self._send( "++eos 3")                   # 0 – CR+LF  # 1 – C    # 2 – LF   # 3 – None
        self._send( "++ifc")                     # Send GPIB reset

    # --- Init
    def _init_adapter(self):
        self._connect_to_adapter()
        self._setup_as_controller()

    # --- Useable in Device worker
    def write(self, cmd):
        self._send(cmd)

    def query(self, cmd, buffer_size=1024*1024):
        self.write(cmd)
        return self.read(buffer_size)

    def read(self, num_bytes=1024):
        self._send('++read eoi')
        return self._recv(num_bytes)
    
    # --- Useable in Adapter Worker
    def unlock(self):
        pass

    # TODO check pyvisa for their TCP socket lock
    def lock(self):
        pass

    def close(self):
        self._socket.close()

    def shutdown(self):
        self._socket.shutdown(SHUT_RDWR) # Further sends and receives are disallowed.

        

