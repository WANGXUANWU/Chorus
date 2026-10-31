"""Unbuffered positioned reads that bypass the operating-system file cache.

On Windows the file is opened with FILE_FLAG_NO_BUFFERING; reads must start at sector
boundaries, cover whole sectors and land in sector-aligned memory, so each request reads
the aligned superset of the wanted byte range into a page-aligned buffer and returns a
zero-copy view of the wanted bytes. On other systems O_DIRECT plays the same role. When
neither is available the reader falls back to ordinary buffered reads (flagged).
"""
import ctypes
import os
import sys
import numpy as np

SECTOR = 4096
MAX_REQ = 64 << 20


class _Buffer:
    """Page-aligned buffer (VirtualAlloc on Windows, over-allocated numpy elsewhere)."""

    def __init__(self, nbytes):
        self.nbytes = int(-(-nbytes // SECTOR) * SECTOR)
        if sys.platform == 'win32':
            k32 = ctypes.WinDLL('kernel32', use_last_error=True)
            k32.VirtualAlloc.restype = ctypes.c_void_p
            k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
            k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
            self._k32 = k32
            self.ptr = k32.VirtualAlloc(None, self.nbytes, 0x3000, 0x04)    # commit|reserve, read/write
            if not self.ptr:
                raise MemoryError('VirtualAlloc failed')
            self.array = np.ctypeslib.as_array((ctypes.c_uint8 * self.nbytes).from_address(self.ptr))
        else:
            raw = np.empty(self.nbytes + SECTOR, dtype=np.uint8)
            off = (-raw.ctypes.data) % SECTOR
            self._raw = raw
            self.array = raw[off:off + self.nbytes]
            self.ptr = self.array.ctypes.data

    def free(self):
        if sys.platform == 'win32' and getattr(self, 'ptr', None):
            self.array = None
            self._k32.VirtualFree(ctypes.c_void_p(self.ptr), 0, 0x8000)
            self.ptr = None


class _OVERLAPPED(ctypes.Structure):
    _fields_ = [('Internal', ctypes.c_size_t), ('InternalHigh', ctypes.c_size_t),
                ('Offset', ctypes.c_ulong), ('OffsetHigh', ctypes.c_ulong), ('hEvent', ctypes.c_void_p)]


class UnbufferedFile:
    """Positioned reads of byte ranges into reusable aligned buffers.

    read(offset, nbytes, slot) returns a uint8 view of exactly the requested bytes; the view
    stays valid until the same slot is read again. issued_bytes counts sector-aligned bytes
    actually requested from the device."""

    def __init__(self, path, buffered_fallback=False):
        self.path = path
        self.issued_bytes = 0
        self.buffers = {}
        self.unbuffered = False
        self._fh = None
        if sys.platform == 'win32' and not buffered_fallback:
            k32 = ctypes.WinDLL('kernel32', use_last_error=True)
            k32.CreateFileW.restype = ctypes.c_void_p
            k32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p,
                                        ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
            k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
                                     ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(_OVERLAPPED)]
            k32.CloseHandle.argtypes = [ctypes.c_void_p]
            h = k32.CreateFileW(os.path.abspath(path), 0x80000000, 0x1, None, 3, 0x20000000 | 0x08000000, None)
            if h is not None and h != ctypes.c_void_p(-1).value:
                self._k32, self._h, self.unbuffered = k32, h, True
        elif hasattr(os, 'O_DIRECT') and not buffered_fallback:
            try:
                self._fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                self.unbuffered = True
            except OSError:
                self._fd = None
        if not self.unbuffered:
            self._fh = open(path, 'rb', buffering=0)
        self.size = os.path.getsize(path)

    def _buffer(self, slot, nbytes):
        b = self.buffers.get(slot)
        if b is None or b.nbytes < nbytes:
            if b is not None:
                b.free()
            b = _Buffer(max(nbytes, 1 << 20))
            self.buffers[slot] = b
        return b

    def read(self, offset, nbytes, slot=0):
        offset, nbytes = int(offset), int(nbytes)
        if not self.unbuffered:
            self._fh.seek(offset)
            data = np.frombuffer(self._fh.read(nbytes), dtype=np.uint8)
            self.issued_bytes += nbytes
            return data
        start = offset // SECTOR * SECTOR
        end = min(-(-(offset + nbytes) // SECTOR) * SECTOR, -(-self.size // SECTOR) * SECTOR)
        span = end - start
        buf = self._buffer(slot, span)
        pos = 0
        while pos < span:
            req = min(MAX_REQ, span - pos)
            if sys.platform == 'win32':
                ov = _OVERLAPPED()
                o = start + pos
                ov.Offset, ov.OffsetHigh = o & 0xFFFFFFFF, o >> 32
                got = ctypes.c_ulong(0)
                ok = self._k32.ReadFile(self._h, ctypes.c_void_p(buf.ptr + pos), req, ctypes.byref(got),
                                        ctypes.byref(ov))
                if not ok:
                    raise OSError(ctypes.get_last_error(), 'ReadFile failed')
                n = got.value
            else:
                n = os.preadv(self._fd, [memoryview(buf.array[pos:pos + req])], start + pos)
            self.issued_bytes += n
            if n < req:          # end of file
                break
            pos += n
        return buf.array[offset - start:offset - start + nbytes]

    def close(self):
        for b in self.buffers.values():
            b.free()
        self.buffers = {}
        if self.unbuffered and sys.platform == 'win32' and getattr(self, '_h', None):
            self._k32.CloseHandle(self._h)
            self._h = None
        elif getattr(self, '_fd', None) is not None:
            os.close(self._fd)
            self._fd = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None
