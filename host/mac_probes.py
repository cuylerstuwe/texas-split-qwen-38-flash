"""Standalone macOS probes, extracted unchanged from the installed Texas probes. No MTPLX runtime import."""

from __future__ import annotations

import ctypes, ctypes.util, functools, threading, time, sys

from dataclasses import dataclass

from typing import Any, Callable

_HOST_PORT_LOCK = threading.Lock()

_HOST_PORT = None

_HOST_VM_INFO64 = 4

@dataclass(frozen=True)
class SystemMemory:
    """One reading of the kernel's memory accounting.

    ``available_bytes`` is the supply: what the kernel can hand out without
    compressing or swapping (free, purgeable and file-backed pages).
    ``free_bytes`` is the part that needs no reclaim at all (free, which
    includes speculative, plus purgeable). ``level_percent`` is
    ``kern.memorystatus_level``, reported for comparison only.
    """

    available_bytes: int
    total_bytes: int
    level_percent: int
    free_bytes: int | None = None
    file_backed_bytes: int | None = None
    wired_bytes: int = 0
    compressor_bytes: int | None = None
    swap_used_bytes: int | None = None
    monotonic_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "available_bytes": int(self.available_bytes),
            "free_bytes": None if self.free_bytes is None else int(self.free_bytes),
            "file_backed_bytes": (
                None if self.file_backed_bytes is None else int(self.file_backed_bytes)
            ),
            "wired_bytes": int(self.wired_bytes),
            "compressor_bytes": (
                None if self.compressor_bytes is None else int(self.compressor_bytes)
            ),
            "swap_used_bytes": (
                None if self.swap_used_bytes is None else int(self.swap_used_bytes)
            ),
            "total_bytes": int(self.total_bytes),
            "memorystatus_level": int(self.level_percent),
        }

@functools.cache
def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.mach_host_self.restype = ctypes.c_uint32
    return libc

def _host_port() -> int:
    # One send right for the process's lifetime: every mach_host_self() call
    # adds a user reference to the host port, and the guard reads the machine
    # every few seconds for as long as the daemon runs. The lock keeps two
    # first calls on different threads from taking two.
    global _HOST_PORT
    with _HOST_PORT_LOCK:
        if _HOST_PORT is None:
            _HOST_PORT = int(_libc().mach_host_self())
        return _HOST_PORT

def _sysctl_int(name: bytes, width: int) -> int | None:
    value = ctypes.c_uint64(0) if width == 8 else ctypes.c_int32(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    rc = _libc().sysctlbyname(name, ctypes.byref(value), ctypes.byref(size), None, 0)
    if rc != 0:
        return None
    return int(value.value)

class _VMStatistics64(ctypes.Structure):
    """``struct vm_statistics64`` from ``<mach/vm_statistics.h>`` (rev2, 160 bytes).

    ``host_statistics64`` is told the buffer size in 32-bit words, so a
    shorter struct is filled only as far as it reaches, never past its end.
    """

    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
        ("swapped_count", ctypes.c_uint64),
    ]

class _XswUsage(ctypes.Structure):
    """``struct xsw_usage`` behind ``sysctl vm.swapusage``."""

    _fields_ = [
        ("xsu_total", ctypes.c_uint64),
        ("xsu_avail", ctypes.c_uint64),
        ("xsu_used", ctypes.c_uint64),
        ("xsu_pagesize", ctypes.c_uint32),
        ("xsu_encrypted", ctypes.c_int32),
    ]

def _read_vm_statistics() -> tuple[_VMStatistics64, int] | None:
    libc = _libc()
    host = _host_port()
    info = _VMStatistics64()
    count = ctypes.c_uint32(ctypes.sizeof(_VMStatistics64) // 4)
    rc = libc.host_statistics64(
        ctypes.c_uint32(host),
        _HOST_VM_INFO64,
        ctypes.byref(info),
        ctypes.byref(count),
    )
    if rc != 0:
        return None
    page = ctypes.c_size_t(0)
    if libc.host_page_size(ctypes.c_uint32(host), ctypes.byref(page)) != 0:
        return None
    if int(page.value) <= 0:
        return None
    return info, int(page.value)

def _read_swap_used_bytes() -> int | None:
    usage = _XswUsage()
    size = ctypes.c_size_t(ctypes.sizeof(usage))
    rc = _libc().sysctlbyname(
        b"vm.swapusage", ctypes.byref(usage), ctypes.byref(size), None, 0
    )
    if rc != 0:
        return None
    return int(usage.xsu_used)

def _supply_from_statistics(info: _VMStatistics64, page: int) -> tuple[int, int]:
    """(free_bytes, file_backed_bytes) from one ``vm_statistics64`` reading.

    ``free_count`` already includes the speculative pages, and so does
    ``external_page_count`` (a speculative page is a file page read ahead):
    the file-backed credit leaves them out so they count once.
    """

    speculative = int(info.speculative_count)
    free = (int(info.free_count) + int(info.purgeable_count)) * int(page)
    file_backed = max(0, int(info.external_page_count) - speculative) * int(page)
    return free, file_backed

def _read_kernel() -> SystemMemory | None:
    total = _sysctl_int(b"hw.memsize", 8)
    if total is None or total <= 0:
        return None
    stats = _read_vm_statistics()
    if stats is None:
        return None
    info, page = stats
    free, file_backed = _supply_from_statistics(info, page)
    level = _sysctl_int(b"kern.memorystatus_level", 4)
    return SystemMemory(
        available_bytes=min(int(total), free + file_backed),
        total_bytes=int(total),
        level_percent=int(level) if level is not None and 0 <= level <= 100 else -1,
        free_bytes=free,
        file_backed_bytes=file_backed,
        wired_bytes=int(info.wire_count) * page,
        compressor_bytes=int(info.compressor_page_count) * page,
        swap_used_bytes=_read_swap_used_bytes(),
        monotonic_s=time.monotonic(),
    )

THERMAL_PRESSURE_LEVELS = ("nominal", "moderate", "heavy", "trapping", "sleeping")

def level_name(level: int) -> str:
    level = int(level)
    if 0 <= level < len(THERMAL_PRESSURE_LEVELS):
        return THERMAL_PRESSURE_LEVELS[level]
    return f"level_{level}"

def notify_thermal_pressure_reader() -> Callable[[], int] | None:
    """A function that reads the current pressure level, or None where the
    notification is unavailable (not macOS, or the lookup failed)."""

    if sys.platform != "darwin":
        return None
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("System") or "/usr/lib/libSystem.B.dylib")
        name = ctypes.c_char_p.in_dll(lib, "kOSThermalNotificationPressureLevelName").value
        register = lib.notify_register_check
        register.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        register.restype = ctypes.c_uint32
        get_state = lib.notify_get_state
        get_state.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
        get_state.restype = ctypes.c_uint32
    except (OSError, AttributeError, ValueError):
        return None
    if not name:
        return None
    token = ctypes.c_int(0)
    if register(name, ctypes.byref(token)) != 0:
        return None

    def read() -> int:
        state = ctypes.c_uint64(0)
        status = get_state(token, ctypes.byref(state))
        if status != 0:
            raise OSError(f"notify_get_state returned {status}")
        return int(state.value)

    return read
