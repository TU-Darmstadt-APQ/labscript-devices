from blacs.tab_base_classes import Worker
from labscript.labscript import LabscriptError
import socket
from enum import IntEnum, auto,Enum

# ------------------------------------------------------------------------------
#                               StreamFramer
# ------------------------------------------------------------------------------
# https://pypi.org/project/streamframer/

from dataclasses import dataclass
from typing import Callable, Tuple, Union
import time

'''
The BufferManager: (Storage and Mutation)
    - When bytes arrive, they are appended to a per-socket buffer.
    - Bytes are never removed automatically.
    - Partial messages survive across recv() calls.
    - Bytes are removed only when consume() is explicitly called.

The Rule:
    - After bytes arrive we ask:
      “Looking at the current buffer, can we identify ONE complete message?”
    - The rule answers:
        * where scanning should resume next time
        * whether more bytes are needed
        * which bytes are the payload
        * how many bytes belong to the message (payload + delimiter)

The Frame: (View + LifeCycle)
    - When the rule finds a match, we create a Frame (a view into the buffer).
    - The buffer must not be appended to while the Frame exists.
    - consume() explicitly discards exactly one message.

Mental Model:
    TCP delivers letters.
    The buffer is the page.
    The rule finds sentences.
    The frame highlights one sentence to read.
    consume() removes that sentence so the next one can be found.
'''


# -------------------------------------------------------
#         Framing rule result contract (INTERNAL)
# -------------------------------------------------------
'''
Defines how a framing rule reports its decision after looking at the buffer.
A rule either says “no full message yet” (_NeedMore) or “one full message found” (_Match).

These are internal by default; users should not depend on them directly.
'''

@dataclass(frozen=True)
class _NeedMore:
    """No complete message yet."""
    next_scan: int  # where scanning should resume when more bytes arrive (inclusive index)

    def __post_init__(self) -> None:
        if self.next_scan < 0:
            raise ValueError("next_scan must be >= 0")


@dataclass(frozen=True)
class _Match:
    """Exactly one complete message exists in the buffer."""
    msg_end: int        # end index (exclusive) of payload
    consume_upto: int   # total bytes to discard when consumed (payload + framing) (exclusive idx)
    next_scan: int = 0  # scan restart position after consumption

    def __post_init__(self) -> None:
        if self.msg_end < 0 or self.consume_upto < 0 or self.next_scan < 0:
            raise ValueError("all fields must be >= 0")
        if not (self.msg_end <= self.consume_upto):
            raise ValueError("msg_end <= consume_upto")


RuleResult = Union[_Match, _NeedMore]                  # result of a framing decision
RuleFn     = Callable[[bytearray, int], RuleResult]   # framing rule signature


# -------------------------------------------------------
#          Delimiter-based framing rule (PUBLIC)
# -------------------------------------------------------
'''
Creates a rule that looks for a specific delimiter in the incoming bytes.
If the delimiter is not found yet, the rule asks for more data. (_NeedMore)
If the delimiter is found, the rule reports one complete message. (_Match)
'''

def until_delim(delim: bytes, *, strip_before: bytes = b"") -> RuleFn:
    # build a rule that detects messages ending with a fixed delimiter
    if not isinstance(delim, (bytes, bytearray)) or len(delim) == 0:
        raise ValueError("delim must be non-empty bytes")
    if not isinstance(strip_before, (bytes, bytearray)):
        raise ValueError("strip_before must be bytes/bytearray")

    delim = bytes(delim)          # ensure immutable delimiter
    strip_before = bytes(strip_before)
    dlen = len(delim)             # delimiter length (for boundary overlap)
    slen = len(strip_before)      # optional bytes to strip from payload

    def rule(buf: bytearray, scan_from: int) -> RuleResult:
        i = buf.find(delim, scan_from)  # search for delimiter in buffer
        if i == -1:
            # delimiter not found: request more bytes, avoid rescanning old data except for last dlen-1 bytes
            return _NeedMore(next_scan=max(0, len(buf) - dlen + 1))

        msg_end = i
        # optionally strip bytes immediately before delimiter (e.g. CR before LF)
        if slen and msg_end >= slen and buf[msg_end - slen:msg_end] == strip_before:
            msg_end -= slen

        # delimiter found: report one complete message
        return _Match(msg_end=msg_end, consume_upto=i + dlen, next_scan=0)

    return rule


def until_eot(eot_byte: int) -> RuleFn:
    """Frame until a single EOT byte is encountered."""
    if not (0 <= eot_byte <= 255):
        raise ValueError("eot_byte must be 0..255")
    return until_delim(bytes([eot_byte]))

# -------------------------------------------------------
#          Frame view and consumption (PUBLIC)
# -------------------------------------------------------
'''
Represents one complete message found in the buffer, without copying bytes.
The frame exposes the payload and deletes it from the buffer only when consume() is called.
'''
@dataclass()
class Frame:
    _view: memoryview                    # zero-copy view of the payload
    _consume_upto: int                   # bytes to discard when consumed
    _on_consume: Callable[[int], None]   # callback into BufferManager
    _consumed: bool = False              # guard against double-consume

    def __post_init__(self) -> None:
        if self._consume_upto < 0:
            raise ValueError("_consume_upto must be >= 0")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.consume()
        return False

    @property
    def view(self) -> memoryview:
        if self._consumed:
            raise RuntimeError("Frame was consumed; view is no longer valid")
        return self._view

    def to_bytes(self) -> bytes:
        if self._consumed:
            raise RuntimeError("Frame was consumed; payload is no longer available")
        return self._view.tobytes()

    def consume(self) -> None:
        if self._consumed:
            return
        try:
            self._view.release()
        except BufferError as e:
            raise RuntimeError(
                "Cannot consume: Frame.view is still exported (someone is holding a memoryview/slice). "
                "Drop all references to Frame.view (and any slices of it) before consuming."
            ) from e
        self._on_consume(self._consume_upto)
        self._consumed = True


# -------------------------------------------------------
#        Per-stream buffer state manager (PUBLIC)
# -------------------------------------------------------
@dataclass()
class _BufState:
    buf: bytearray
    scan: int = 0  # inclusive start index for the next scan


class BufferManager:
    """Stores buffer + scan position for exactly one byte stream."""
    def __init__(self) -> None:
        self._st = _BufState(buf=bytearray(), scan=0)
        self._frame_outstanding = False

    def assert_can_mutate(self) -> None:
        if self._frame_outstanding:
            raise RuntimeError("Buffer cannot be mutated while a Frame is outstanding")

    def buf_and_scan(self) -> Tuple[bytearray, int]:
        return self._st.buf, self._st.scan

    def set_scan(self, scan: int) -> None:
        self._st.scan = max(0, scan)

    def mark_frame_outstanding(self) -> None:
        if self._frame_outstanding:
            raise RuntimeError("Frame already outstanding; call Frame.consume() before reading again")
        self._frame_outstanding = True

    def clear_frame_outstanding(self) -> None:
        self._frame_outstanding = False

    def consume(self, n: int) -> None:
        buf = self._st.buf

        if n <= 0:
            return

        if n >= len(buf):
            buf.clear()
            self._st.scan = 0
            self._frame_outstanding = False
            return

        del buf[:n]
        self._st.scan = 0
        self._frame_outstanding = False

    def clear(self) -> None:
        """Hard reset buffer + scan (useful before probing)."""
        self._st.buf.clear()
        self._st.scan = 0
        self._frame_outstanding = False


# -------------------------------------------------------
#      Read one framed message from a stream (PUBLIC)
# -------------------------------------------------------
def read_with_rule(
    sock,
    mgr: BufferManager,
    rule: RuleFn,
    *,
    max_bytes: int = 1024 * 1024,
    recv_size: int = 4096,
    overall_timeout: float = 2.0,
    per_recv_timeout: float = 0.3,
) -> Frame:
    """
    Read until rule identifies exactly one complete message, or until overall_timeout expires.

    - per_recv_timeout: socket timeout for each recv() call (short poll interval)
    - overall_timeout: total time budget to obtain a complete frame
    """
    if not callable(rule):
        raise TypeError("rule must be callable")
    if per_recv_timeout <= 0 or per_recv_timeout > 3.0:
        raise ValueError("per_recv_timeout must be > 0 and <= 3.0")
    if overall_timeout <= 0:
        raise ValueError("overall_timeout must be > 0")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if recv_size <= 0:
        raise ValueError("recv_size must be > 0")

    mgr.assert_can_mutate()
    buf, scan = mgr.buf_and_scan()
    deadline = time.monotonic() + overall_timeout

    last_buf_len = -1
    last_scan = -1

    old_timeout = None
    try:
        old_timeout = sock.gettimeout()
        sock.settimeout(per_recv_timeout)

        while True:
            if time.monotonic() >= deadline:
                raise RuntimeError("overall_timeout expired while waiting for framed message")

            if len(buf) != last_buf_len or scan != last_scan:
                last_buf_len = len(buf)
                last_scan = scan

                res = rule(buf, scan)
                scan = res.next_scan
                mgr.set_scan(scan)

                if isinstance(res, _Match):
                    mv = memoryview(buf)[:res.msg_end]
                    mgr.mark_frame_outstanding()
                    return Frame(
                        _view=mv,
                        _consume_upto=res.consume_upto,
                        _on_consume=mgr.consume,
                    )

            try:
                chunk = sock.recv(recv_size)
            except socket.timeout:
                continue

            if not chunk:
                raise RuntimeError("Socket closed")

            if len(buf) + len(chunk) > max_bytes:
                raise RuntimeError("Exceeded max_bytes (missing terminator?)")

            buf.extend(chunk)

            if scan > len(buf):
                scan = len(buf)
            mgr.set_scan(scan)

    finally:
        try:
            sock.settimeout(old_timeout)
        except Exception:
            pass


# -------------------------------------------------------
#       Read whatever arrives until timeout (PUBLIC)
# -------------------------------------------------------
def read_until_timeout(
    sock,
    mgr: BufferManager,
    *,
    max_bytes: int = 1024 * 1024,
    max_total_buf: int = 4 * 1024 * 1024,
    overall_timeout: float = 2.0,
    per_recv_timeout: float = 0.3,
    consume: bool = False,
) -> bytes:
    """
    Read whatever arrives on the socket until overall_timeout expires
    or max_bytes new bytes have been received.

    Returns ONLY bytes that arrived during this call (not previously buffered bytes).

    Buffer behavior:
    - consume=False: previously buffered bytes remain untouched; new bytes appended.
    - consume=True: consumes only bytes received during this call; old buffered bytes remain.

    Safety limits:
    - max_bytes limits how many new bytes this call may receive.
    - max_total_buf limits total buffered bytes (old + new).
    """
    if per_recv_timeout <= 0 or per_recv_timeout > 3.0:
        raise ValueError("per_recv_timeout must be > 0 and <= 3.0")
    if overall_timeout <= 0:
        raise ValueError("overall_timeout must be > 0")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if max_total_buf <= 0:
        raise ValueError("max_total_buf must be > 0")

    mgr.assert_can_mutate()
    buf, _scan = mgr.buf_and_scan()
    start = len(buf)
    deadline = time.monotonic() + overall_timeout

    old_timeout = None
    try:
        old_timeout = sock.gettimeout()
        sock.settimeout(per_recv_timeout)

        while True:
            if time.monotonic() >= deadline:
                break

            new_len = len(buf) - start
            if new_len >= max_bytes:
                break

            try:
                chunk = sock.recv(max_bytes - new_len)
            except socket.timeout:
                continue

            if not chunk:
                break

            if len(buf) + len(chunk) > max_total_buf:
                raise RuntimeError("Buffer exceeded max_total_buf; caller must consume/clear")

            buf.extend(chunk)

    finally:
        try:
            sock.settimeout(old_timeout)
        except Exception:
            pass

        if consume:
            mgr.consume(len(buf) - start)

    return bytes(buf[start:])


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