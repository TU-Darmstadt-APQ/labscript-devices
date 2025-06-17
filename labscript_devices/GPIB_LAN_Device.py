from blacs.tab_base_classes import Worker
from labscript.labscript import LabscriptError
from socket import socket,error , AF_INET, SOCK_STREAM


class GPIB_LAN_Worker(Worker):

    PORT = 1234

    def init(self):
        global h5py
        import labscript_utils.h5_lock
        import h5py

        # --------------------------------- Connecting and Initialize Adapter 
        self.socket  = socket(AF_INET,SOCK_STREAM)

        # --- Attributes
        self.timeout = self._set_timeout(1)              # Socket communication timeout

        self._init_GPIB()


    def _set_timeout(self, value):   # https://prologix.biz/downloads/PrologixGpibEthernetManual.pdf#page=13
        if value < 1e-3 or value > 3:
            raise LabscriptError('❌ Timeout must be >= 1e-3 (1ms) and <= 3 (3s)')
        self._timeout = value
        self.socket.settimeout(value)

    # --- Send and Receive socket
    def _send(self, cmd):
        self.socket.sendall((cmd + '\n').encode('ascii'))

    def _recv(self, byte_num=1024):
        value = self.socket.recv(byte_num)
        return value.decode('ascii')
    
    # --- Connect Adapter
    def _connect_to_adapter(self):
        try :
            self.socket.connect((self.ip_adapter, self.PORT))
            print(f"✅ Kofotronic Adpater : {self.ip_adapter}")
        except Exception:
            raise LabscriptError("Connection to Adapter not successful")
        
    # -- Connect to Instrument
    def _connect_instr(self):
        try:
            self._send(f'++addr {self.GPIB_address}')
            print(f"✅ {self.device_name}")
        except Exception:
            raise LabscriptError("Connection to Instrument not successful")


    # --- Setup
    def setup_as_controller(self):
        self._send( "++mode 1")                  # 0 Device Mode         # 1 Controller
        self._connect_instr()
        self._send( "++auto 0")                  # 0 Instrument LISTEN   # 1 Instrument TALK.
        self._send( "++eos 3")                   # 0 – CR+LF  # 1 – C    # 2 – LF   # 3 – None
        self._send( "++ifc")                     # Send GPIB reset

    # --- cp
    def _init_GPIB(self):
        self._connect_to_adapter()
        self.setup_as_controller()
        return
    
    def shutdown(self):
        if self.GPIB_connection is not None:
            try:
                self.GPIB_connection.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass 
            self.GPIB_connection.close()
            self.GPIB_connection = None

    def abort_transition_to_buffered(self):
        return self.transition_to_manual()

    def abort_buffered(self):
        return self.transition_to_manual()

    def transition_to_manual(self):
        return True
    

    #######################################################################################
    #                             Instrument related methodes                             #
    ####################################################################################### 

    def write(self, cmd):
        self._send(cmd)

    def query(self, cmd, buffer_size=1024*1024):
        self.write(cmd)
        return self.read(buffer_size)

    def read(self, num_bytes=102400):
        self._send('++read eoi')
        return self._recv(num_bytes)

