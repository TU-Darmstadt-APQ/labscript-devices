from blacs.tab_base_classes import Worker
from labscript.labscript import LabscriptError
import socket
import time
from typing import Optional, Tuple
from .streamframer import *
from enum import IntEnum, auto,Enum


# ------------------------------------------------------------------------------
#                               READ STRATEGY
# ------------------------------------------------------------------------------
# This enum describes HOW the PC decides that an instrument reply is complete.
#
# IMPORTANT TERMINOLOGY (Prologix / GPIB):
#
# EOS (End Of String):
#   Characters appended when sending data to the instrument to match the instrument’s required line terminator.
#   Examples: LF, CR, CR+LF, or None. (see ENUM EosStrategy )
#
# EOI (End Or Identify):
#   A GPIB BUS signal asserted on the LAST BYTE of a transfer.
#   It marks the final byte of a command or a reply on the GPIB side. (Support by most intrument BUT NOT ALL)
#
# EOT (End Of Transmission):
#   A FAKE byte added by the ADAPTER to the TCP stream (adapter → PC only).
#   It exists solely to let the PC detect EOI over TCP.
#
# IN SHORT:
#                EOI 
#                 ↓   
#   Instrument → GPIB → Adapter → TCP → PC
#       ↑                 ↑
#      EOS                EOT (Only works with EOI)
#
# ------------------------------------------------------------------------------
class ReadStrategy(Enum):
    """ Strategy used by the PC to determine when a read() is complete."""

    EOT = auto()
    # Read until the adapter appends the configured EOT byte.
    # Fast and precise, requires EOI support.

    LINE = auto()   
    # Read until a line delimiter (LF or CR+LF).
    # Typical for Prologix command responses (e.g. ++ver).
    # Works even if adapter ignores EOI

    TIMEOUT = auto()
    # Read until no more bytes arrive for a fixed timeout.
    # Slow but works even without EOI/EOT support.

# ------------------------------------------------------------------------------
#                               EOS STRATEGY
# ------------------------------------------------------------------------------
# This enum describes HOW the instrument decides when a string is
# considered to be terminated (EOS).
#
# IMPORTANT TERMINOLOGY:
#
# EOS (End Of String):
#   Characters appended by the ADAPTER when SENDING data TO the instrument.
#   Examples include LF, CR, CR+LF, or None.
#
# ------------------------------------------------------------------------------
class EosStrategy(IntEnum):
    """ Strategy used by the instrument to append EOS characters. """
    CRLF = 0   # Carriage Return + Line Feed
    CR   = 1   # Carriage Return
    LF   = 2   # Line Feed 
    NONE = 3   # No EOS

#  ----------------------------------------------------------------------------------
#                                   Worker CLASS
# -----------------------------------------------------------------------------------
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
                self.GPIB_connection = AdapterConnection(self.device_name , ip_add, self.eos_strategy , gpib_add)
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
    

#  ----------------------------------------------------------------------------------
#                                   ADAPTER CLASS
# -----------------------------------------------------------------------------------
class AdapterConnection:
    """
    Connection class for a Kofotronic (Prologix-like) adapter that emulates a GPIB connection for PyVISA.
    Although multiple read strategies are implemented (Not Tested), only the line-based read strategy is used,
    as it is the most reliable across instruments.
    """
    PORT = 1234
    def __init__(self, device_name, ip_ad, eos_strategy:str , gpib_ad,timeout=1):

        # --------------------------------- Labscript
        self.device_name        = device_name
        self.ip_ad              = ip_ad
        self.gpib_ad            = gpib_ad 
                                      
        # --------------------------------- Communication
        # EOI/EOS/EOT
        self._eos_strategy      = eos_strategy                              # Instrument End of String
        self._eot_char          = None
         
        # RULES
        self._prologix_rule     = until_delim(b"\n", strip_before=b"\r")    # Read Rule for prologix like adapters
        self._instr_line_rule   = None                                      # Read LINE Rule 
        self._eot_rule          = None                                      # Read EOT Rule

        # BUFFER AND READ 
        self.buf_man            = BufferManager()
        self._read_strategy     = ReadStrategy.LINE                         # Reliable Read Strategy
        self._gpib_read_impl    = self._select_gpib_read_impl()

        # --------------------------------- Adapter Connection
        self._socket  = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.set_timeout(timeout)                                           # Socket communication timeout
        self._init_adapter()

    # --- The Used Read Function
    def _select_gpib_read_impl(self):
        rs = self._read_strategy
        if rs is ReadStrategy.LINE: # EOS-based read (LF/CR/CRLF), reliable 
            self._instr_line_rule = until_delim(b"\n", strip_before=b"\r")
            self._eot_rule = None
            self._eot_char = None
            return self._gpib_read_line
        if rs is ReadStrategy.EOT: # GPIB END/EOI-based read, fastest when supported by instrument
            self._eot_char = 0
            self._eot_rule = until_eot(self._eot_char)
            self._instr_line_rule = None
            return self._gpib_read_eot
        if rs is ReadStrategy.TIMEOUT: # Timeout-based read, fallback when no termination is available
            self._instr_line_rule = None
            self._eot_rule = None
            self._eot_char = None
            return self._gpib_read_timeout
        raise LabscriptError(f"Unhandled read strategy: {rs}")
        
    # --- Set Socket Timeout
    def set_timeout(self, value):   # https://prologix.biz/downloads/PrologixGpibEthernetManual.pdf#page=13
        'Timeout must be >= 1e-3 (1ms) and <= 3 (3s)'
        if value < 1e-3 or value > 3:
            raise LabscriptError('🚫 Timeout must be >= 1e-3 (1ms) and <= 3 (3s)')
        self._socket.settimeout(value)

    # --- Send Command over Socket to prologix adpater (Used for Prologix Commands & Instrument Commands)
    def _send_to_adap(self, cmd):
        try:
            encoded_value = (f"{cmd}\n").encode('ascii')    # !!! the \n is the adapter terminator
            self._socket.sendall(encoded_value)
        except Exception as e:
            raise LabscriptError(f" 🚫 Failed to send command :{e}")
        
    # --- Read Functions for Prologix Commands  
    def _read_prologix_line(self) -> str:
        # Prologix responses are line-based; accept \n or \r\n
        f = read_with_rule(self._socket, self.buf_man, self._prologix_rule)
        try:
            raw = f.to_bytes()
        finally:
            f.consume()
        return raw.decode("ascii", errors="replace").strip()
    
    def _read_prologix_cmds(self, max_empty: int = 50) -> str:
        empties = 0
        while True:
            line = self._read_prologix_line()
            if line:
                return line
            empties += 1
            if empties >= max_empty:
                raise LabscriptError("Too many empty Prologix lines; giving up")

    # --- Read Functions for Instrument
    def _gpib_read_eot(self, max_bytes, overall_timeout):
        if self._eot_rule is None:
            raise LabscriptError("EOT strategy selected but _eot_rule is not initialized")
        self._send_to_adap("++read eoi")
        f = read_with_rule( self._socket, self.buf_man, self._eot_rule, max_bytes=max_bytes, overall_timeout=overall_timeout)
        try:
            raw = f.to_bytes()
        finally:
            f.consume()
        return raw.decode("ascii", errors="replace")

    def _gpib_read_line(self, max_bytes, overall_timeout):
        self._send_to_adap("++read")
        f = read_with_rule(
            self._socket, self.buf_man, self._instr_line_rule,
            max_bytes=max_bytes, overall_timeout=overall_timeout
        )
        try:
            raw = f.to_bytes()
        finally:
            f.consume()
        return raw.decode("ascii", errors="replace").strip()

    def _gpib_read_timeout(self, max_bytes, overall_timeout):
        # Returns ONLY bytes that arrive during this call.
        # consume=True discards ONLY those new bytes; old buffered bytes remain.
        self._send_to_adap("++read")
        raw = read_until_timeout(self._socket, self.buf_man, consume=True, max_bytes=max_bytes, overall_timeout=overall_timeout)
        return raw.decode("ascii", errors="replace")

    def _read_gpib(self, max_bytes=1024*1024, overall_timeout=2.0) -> str:
        return self._gpib_read_impl(max_bytes=max_bytes, overall_timeout=overall_timeout)

    # --- Connect Adapter
    def _connect_to_adapter(self):
        try :
            self._socket.connect((self.ip_ad, self.PORT))
            version = self._get_version()
            print(f"✅ Adpater : {self.ip_ad} - Version :{version}")
        except Exception as e:
            raise LabscriptError(f"🚫 Connection to Adapter not successful :{e}")
        
    # -- Connect to Instrument
    def _connect_instr(self):
        try:
            self._set_gpib_address(self.gpib_ad)
            print(f"✅ Device : {self.device_name}")
        except Exception as e:
            raise LabscriptError(f"🚫 Connection to Instrument not successful :{e}")
        
    # --- Useable in Device worker
    def write(self, cmd):
        self._send_to_adap(cmd)

    def read(self, max_bytes=1024*1024, overall_timeout=2.0):
        return self._read_gpib(max_bytes=max_bytes, overall_timeout=overall_timeout)

    def query(self, cmd: str, max_bytes=1024*1024, overall_timeout=2.0):
        self.write(cmd)
        return self.read(max_bytes=max_bytes, overall_timeout=overall_timeout)

    # --- Init
    def _init_adapter(self):
        # --- Fix Settings
        self._connect_to_adapter()          # TCP & ++ver
        self._set_mode(1)                   # controller mode
        self._connect_instr()               # ++addr <gpib>
        self._set_auto(0)                   # read-after-write OFF (Preferably)
        # --- CASE 1 : Read By Line         (Reliable) Tested
        self._set_eos(self._eos_strategy)   # LF terminator
        self._set_eot_enable(0)             # LINE: No EOT
        # --- CASE 2 : Read EOI             (Fast) Not Tested
        # self._read_strategy     = ReadStrategy.EOT                          
        # self._eot_char          = 10
        # self._gpib_read_impl    = self._select_gpib_read_impl()
        # self._set_eot_char(self._eot_char)
        # self._set_eoi(1)
        # self._set_eot_enable(1)                                             
        # --- CASE 3 : Read Time             (FallBack) Not Tested
        # self._read_strategy     = ReadStrategy.TIMEOUT                      
        # self._set_eot_enable(0)                                             
        # self._gpib_read_impl    = self._select_gpib_read_impl()
        # --- Test
        # self.write("STS? 1")
        # result = self.read(overall_timeout=2.0)
        # print("===========", result)

        
    # --- Useable in Adapter Worker
    def unlock(self):
        pass

    # TODO check pyvisa for their TCP socket lock
    def lock(self):
        pass

    def close(self):
        self._socket.close()

    def shutdown(self):
        try:
            self._socket.shutdown(socket.SHUT_RDWR) # Further sends and receives are disallowed.
        except OSError:
            pass  # socket already closed or never connected
    # ==========================================================
    #               Prologix Adapter Commands
    # ==========================================================
    
    # --- Helpers 
    def _check_01(self, name: str, value: int): # Some Helper to check 0 or 1
        if value not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1")

    # --- Adapter info
    def _get_version(self):
        """Return the Prologix adapter firmware version string."""
        self._send_to_adap("++ver")
        return self._read_prologix_cmds()

    # --- Adapter mode / addressing
    def _set_mode(self , value: int):
        """Set the Prologix adapter to GPIB device/controller mode (0=device, 1=controller)."""
        self._check_01("mode", value)
        self._send_to_adap(f"++mode {value}")

    def _get_mode(self):
        """Return the current Prologix mode (0=device, 1=controller)."""
        self._send_to_adap("++mode")
        return self._read_prologix_cmds()

    def _set_gpib_address(self, addr: int):
        """Set the active GPIB address for subsequent instrument commands."""
        self._send_to_adap(f"++addr {addr}")
    
    # --- Write termination / bus signaling
    def _set_auto(self, value: int):
        """Set auto read-after-write: 0=off, 1=on."""
        self._check_01("auto", value)
        self._send_to_adap(f"++auto {value}")

    def _set_eos(self, eos: EosStrategy):
        """Configure the end-of-string (EOS) termination mode (0=CR+LF, 1=CR, 2=LF, 3=None)."""
        self._send_to_adap(f"++eos {eos}")

    def _set_eoi(self, value: int):
        """Control GPIB EOI line on writes (0=do not assert, 1=assert on last byte)."""
        self._check_01("eoi", value)
        self._send_to_adap(f"++eoi {value}")

    # --- Read post-processing (host-side)
    def _set_eot_enable(self, value: int):
        """Enable appending an EOT (End Of Transmission) character to read data (0=off, 1=on)."""
        self._check_01("eot_enable", value)
        self._send_to_adap(f"++eot_enable {value}")

    def _set_eot_char(self, value: int):
        if not (0 <= value <= 255):
            raise LabscriptError("eot_char must be 0..255")
        self._send_to_adap(f"++eot_char {value}")
        self._eot_char = value
        self._eot_rule = until_eot(value)
        
    # --- Bus control
    def _interface_clear(self):
        """Issue a GPIB Interface Clear (IFC) to reset the bus."""
        self._send_to_adap("++ifc")

    # --- Status / Polling
    def _get_serial_poll(self):
        """Perform a GPIB serial poll and return the status byte (0-255)."""
        self._send_to_adap("++spoll")
        return self._read_prologix_cmds()

    def _get_srq(self):
        """Return the current SRQ line state (0=no request, 1=request asserted)."""
        self._send_to_adap("++srq")
        return self._read_prologix_cmds()

    # ==========================================================
    #                     Futur Implementation  
    # ==========================================================