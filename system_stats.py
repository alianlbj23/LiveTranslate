"""Numbers for the overlay status block (MonitorBar).

The GUI process does little work itself: speech recognition runs in the ASR worker
subprocess and translation in Ollama (a separate process), so measuring only the GUI
process showed "CPU 0% GPU 0MB". Here CPU is the whole machine, RAM is the LiveTranslate
process tree (GUI + ASR worker), and GPU is the whole card read through NVIDIA NVML
(nvml.dll ships with the driver; called through ctypes, no extra package)."""
import ctypes
import logging
import os
import sys
from typing import NamedTuple, Optional

import psutil

log = logging.getLogger("LiveTranslate.Stats")

NVML_SUCCESS = 0


class GpuStats(NamedTuple):
    used_mb: float
    total_mb: float
    util_percent: Optional[int]  # GPU utilisation %, None when the driver does not report it


def system_cpu_percent() -> float:
    """Whole-machine CPU % since the previous call (the first call returns 0.0)."""
    return psutil.cpu_percent(interval=None)


def process_tree_rss_mb(proc) -> float:
    """RSS of `proc` plus all its descendants (the ASR worker and the venv launcher)."""
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue  # exited between listing and reading (e.g. ASR worker restart)
        except psutil.AccessDenied as e:
            log.debug(f"Cannot read memory of pid {child.pid}: {e}")
    return total / 1024 / 1024


def format_gpu(stats: Optional[GpuStats]) -> str:
    if stats is None:
        return "N/A"
    text = f"{stats.used_mb / 1024:.1f}/{stats.total_mb / 1024:.1f}GB"
    if stats.util_percent is not None:
        text += f" {stats.util_percent}%"
    return text


class _Memory(ctypes.Structure):  # nvmlMemory_t
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _Utilization(ctypes.Structure):  # nvmlUtilization_t
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


def _load_nvml():
    """nvml.dll from the NVIDIA driver, by absolute path (no DLL search-order surprises)."""
    if sys.platform == "win32":
        candidates = [
            os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvml.dll"),
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI", "nvml.dll"),
        ]
    else:
        candidates = ["libnvidia-ml.so.1"]
    for path in candidates:
        if sys.platform == "win32" and not os.path.isfile(path):
            continue
        try:
            return ctypes.CDLL(path)
        except OSError as e:
            log.debug(f"Cannot load {path}: {e}")
    return None


class NvmlGpu:
    """Whole-card memory use and load of one NVIDIA GPU. read() returns None when NVML is
    not available; after the first failure it stops trying (logged once)."""

    def __init__(self, index: int = 0, lib=None):
        self.index = int(index)
        self._lib = lib
        self._handle = None
        self._failed = False

    def _call(self, name, *args):
        rc = getattr(self._lib, name)(*args)
        if rc != NVML_SUCCESS:
            raise RuntimeError(f"{name} returned NVML error {rc}")

    def _open(self):
        if self._lib is None:
            self._lib = _load_nvml()
            if self._lib is None:
                raise RuntimeError("nvml library not found (no NVIDIA driver?)")
        self._call("nvmlInit_v2")
        handle = ctypes.c_void_p()
        self._call("nvmlDeviceGetHandleByIndex_v2", ctypes.c_uint(self.index), ctypes.byref(handle))
        self._handle = handle

    def read(self) -> Optional[GpuStats]:
        if self._failed:
            return None
        try:
            if self._handle is None:
                self._open()
            mem = _Memory()
            self._call("nvmlDeviceGetMemoryInfo", self._handle, ctypes.byref(mem))
            util = _Utilization()
            util_percent = None
            if self._lib.nvmlDeviceGetUtilizationRates(self._handle, ctypes.byref(util)) == NVML_SUCCESS:
                util_percent = int(util.gpu)
            return GpuStats(used_mb=mem.used / 1024 / 1024, total_mb=mem.total / 1024 / 1024, util_percent=util_percent)
        except (OSError, RuntimeError, AttributeError) as e:
            self._failed = True
            log.warning(f"GPU stats unavailable for GPU {self.index}, showing N/A: {e}")
            return None
