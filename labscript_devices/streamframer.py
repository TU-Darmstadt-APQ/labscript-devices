from __future__ import annotations
import socket
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


# INTERNAL: convenience presets (not part of stable API)
_FRAMING_RULES = {
    "CRLF": until_delim(b"\r\n"),                    # strict RFC-style line ending
    "CR":   until_delim(b"\r"),                      # legacy / instrument-specific
    "LINE": until_delim(b"\n", strip_before=b"\r"),  # newline, optional preceding CR
}

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