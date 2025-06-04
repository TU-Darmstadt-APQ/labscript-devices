
from labscript.labscript import LabscriptError
from socket import socket,error , AF_INET, SOCK_STREAM


class GPIBLANAdapterDevice:

    PORT = 1234

    def __init__(self,
                 address,
                 timeout = 1 ,
                 verbose = True
                 ):
        # --------------------------------- Connecting and Initialize Adapter 
        self.socket  = socket(AF_INET,SOCK_STREAM)

        # --- Attributes
        self.verbose = verbose
        self.address = address              # IP adress of the adapter
        self.timeout = timeout              # Socket communication timeout

        # --- Init
        self._connect_to_adapter()
        self.setup_as_controller()

    #######################################################################################
    #                                 Setter & Getters                                    #
    ####################################################################################### 
    # --- timeout
    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):   # https://prologix.biz/downloads/PrologixGpibEthernetManual.pdf#page=13
        if value < 1e-3 or value > 3:
            raise LabscriptError('❌ Timeout must be >= 1e-3 (1ms) and <= 3 (3s)')
        self._timeout = value
        self.socket.settimeout(value)

    #######################################################################################
    #                                    Helpers                                          #
    ####################################################################################### 

    # --- Error Handler Wrapper
    def _handle_error(func, error = error ):
        def wrapper(self, *args, **kwargs):
            try:
                result = func(self, *args, **kwargs)
                if self.verbose:
                    # My OCD is kicking here
                    name = func.__name__[1:] if func.__name__.startswith("_") else func.__name__
                    func_name = name.replace("_", " ")
                    print(f"✅ {func_name}")
                return 
            except error as e:
                LabscriptError(f"❌ Error in {func.__name__}: {e}")
        return wrapper
    
    # --- Connect
    @_handle_error
    def _connect_to_adapter(self):
        self.socket.connect((self.address, self.PORT))

    # --- Send and Receive socket
    def _send(self, cmd):
        self.socket.sendall((cmd + '\n').encode('ascii'))

    def _recv(self, byte_num=1024):
        value = self.socket.recv(byte_num)
        return value.decode('ascii')

    #######################################################################################
    #                             Adapter related Methodes                                #
    ####################################################################################### 
    def set_timeout(self, timeout):
        # https://prologix.biz/downloads/PrologixGpibEthernetManual.pdf#page=13
        if timeout < 1e-3 or timeout > 3:
            raise LabscriptError('❌ Timeout must be >= 1e-3 (1ms) and <= 3 (3s)')
        self.timeout = timeout
        self.socket.settimeout(self.timeout)

    @_handle_error
    def setup_as_controller(self):
        self._send( "++mode 1")                  # 0 Device Mode         # 1 Controller
        self._send( "++auto 0")                  # 0 Instrument LISTEN   # 1 Instrument TALK.
        self._send( "++eos 3")                   # 0 – CR+LF  # 1 – C    # 2 – LF   # 3 – None
        self._send( "++ifc")                     # Send GPIB reset

    def get_address_gpib(self):
        '''The addr command is used to configure, or query the GPIB address'''
        self._send('++addr')  
        return self._recv() 
    
    def set_address_gpib(self, address_gpib):
        '''The addr command is used to configure, or query the GPIB address'''
        self._send(f'++addr {address_gpib}')      

    @_handle_error
    def close(self):
        self.socket.close()

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

