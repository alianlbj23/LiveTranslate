"""Overlay status block (MonitorBar): CPU / RAM / GPU must reflect where the work runs.

Regression 2026-10-20: the bar showed "CPU 0% RAM 738MB GPU 0MB" while subtitles were
running, because it measured only the GUI process. Speech recognition runs in the ASR
worker subprocess (CTranslate2 on the GPU) and translation in Ollama (another process),
so neither psutil on the GUI process nor torch.cuda.memory_allocated() saw any of it.

Now: CPU = whole machine, RAM = LiveTranslate process tree (GUI + ASR worker),
GPU = whole card through NVIDIA NVML (nvml.dll from the driver, read with ctypes)."""
import ctypes
import logging
import os
import subprocess
import sys
import time

import psutil
import pytest

import system_stats
from i18n import t
from subtitle_overlay import MonitorBar


class _FakeGpu:
    def __init__(self, stats):
        self.stats = stats
        self.index = 0

    def read(self):
        return self.stats


def _tree_rss(pid):
    # The venv python.exe is a small launcher that runs the real interpreter as its own
    # child (the ASR worker looks the same), so sum the whole tree.
    p = psutil.Process(pid)
    total = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            total += c.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


def _wait_rss(pid, mb, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        if _tree_rss(pid) >= mb * 1024 * 1024:
            return
        time.sleep(0.1)
    raise AssertionError(f"child never reached {mb} MB")


def test_ram_counts_the_asr_worker_child_process():
    child = subprocess.Popen([sys.executable, "-c",
                              "import time; b = bytearray(150 * 1024 * 1024); time.sleep(60)"])
    try:
        _wait_rss(child.pid, 140)
        bar = MonitorBar()  # first update happens here (the old code imported torch in it)
        own = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        bar._update_system()
        assert bar._ram_mb >= own + 130, (bar._ram_mb, own)
    finally:
        # kill the real interpreter too (the venv launcher's child), not only the launcher
        for proc in psutil.Process(child.pid).children(recursive=True):
            proc.kill()
        child.kill()
        child.wait(timeout=10)


def test_cpu_is_the_whole_machine_and_gpu_the_whole_card(monkeypatch):
    monkeypatch.setattr(system_stats, "system_cpu_percent", lambda: 37.4)
    bar = MonitorBar()
    bar._gpu = _FakeGpu(system_stats.GpuStats(used_mb=8192, total_mb=12288, util_percent=41))
    bar._update_system()
    text = bar._stats_label.text()
    assert "CPU</span> 37%" in text
    assert "8.0/12.0GB 41%" in text
    assert bar._stats_label.toolTip() == t("ovl_stats_tip")


def test_gpu_without_nvml_shows_na():
    bar = MonitorBar()
    bar._gpu = _FakeGpu(None)
    bar._update_system()
    assert "GPU</span> N/A" in bar._stats_label.text()


def test_gpu_index_follows_the_asr_device():
    bar = MonitorBar()
    bar.update_asr_device("Whisper large-v3-turbo [cuda:1]")
    assert bar._gpu.index == 1
    bar.update_asr_device("Whisper large-v3-turbo [cpu]")
    assert bar._gpu.index == 1  # CPU ASR: keep watching the card that was in use


def test_format_gpu():
    assert system_stats.format_gpu(system_stats.GpuStats(used_mb=8601, total_mb=12227, util_percent=7)) == "8.4/11.9GB 7%"
    assert system_stats.format_gpu(system_stats.GpuStats(used_mb=1024, total_mb=12288, util_percent=None)) == "1.0/12.0GB"
    assert system_stats.format_gpu(None) == "N/A"


class _ProcStub:
    def __init__(self, rss, children=(), gone=False):
        self._rss, self._children, self._gone = rss, list(children), gone

    def memory_info(self):
        if self._gone:
            raise psutil.NoSuchProcess(pid=1234)
        return type("M", (), {"rss": self._rss})()

    def children(self, recursive=False):
        return self._children


def test_process_tree_skips_a_child_that_just_exited():
    mb = 1024 * 1024
    tree = _ProcStub(100 * mb, [_ProcStub(50 * mb), _ProcStub(0, gone=True)])
    assert system_stats.process_tree_rss_mb(tree) == pytest.approx(150)


class _FakeNvml:
    """Mimics nvml.dll entry points; out-parameters arrive as ctypes.byref()."""

    def __init__(self, init_rc=0, util_rc=0):
        self.init_rc, self.util_rc, self.init_calls, self.index = init_rc, util_rc, 0, None

    def nvmlInit_v2(self):
        self.init_calls += 1
        return self.init_rc

    def nvmlDeviceGetHandleByIndex_v2(self, index, ref):
        self.index = index.value if hasattr(index, "value") else index
        ref._obj.value = 0x1234
        return 0

    def nvmlDeviceGetMemoryInfo(self, handle, ref):
        ref._obj.total, ref._obj.used, ref._obj.free = 12 * 1024 ** 3, 8 * 1024 ** 3, 4 * 1024 ** 3
        return 0

    def nvmlDeviceGetUtilizationRates(self, handle, ref):
        ref._obj.gpu = 41
        return self.util_rc

    def nvmlDeviceGetName(self, handle, buf, length):
        name = b"NVIDIA GeForce RTX 5070"
        ctypes.memmove(buf, name, len(name))
        return 0


def test_nvml_reader_reports_the_card_name():
    assert system_stats.NvmlGpu(0, lib=_FakeNvml()).name() == "NVIDIA GeForce RTX 5070"
    assert system_stats.NvmlGpu(0, lib=_FakeNvml(init_rc=9)).name() is None


def test_nvml_reader_reports_memory_and_load():
    lib = _FakeNvml()
    stats = system_stats.NvmlGpu(1, lib=lib).read()
    assert stats == system_stats.GpuStats(used_mb=8192, total_mb=12288, util_percent=41)
    assert lib.index == 1


def test_nvml_reader_without_load_still_reports_memory():
    stats = system_stats.NvmlGpu(0, lib=_FakeNvml(util_rc=3)).read()  # NVML_ERROR_NOT_SUPPORTED
    assert stats == system_stats.GpuStats(used_mb=8192, total_mb=12288, util_percent=None)


def test_nvml_failure_returns_none_logs_once_and_stops_retrying(caplog):
    lib = _FakeNvml(init_rc=9)  # NVML_ERROR_DRIVER_NOT_LOADED
    gpu = system_stats.NvmlGpu(0, lib=lib)
    with caplog.at_level(logging.WARNING, logger="LiveTranslate.Stats"):
        assert gpu.read() is None
        assert gpu.read() is None
    assert lib.init_calls == 1
    assert len([r for r in caplog.records if "GPU" in r.getMessage()]) == 1


def test_missing_nvml_library_returns_none(monkeypatch):
    monkeypatch.setattr(system_stats, "_load_nvml", lambda: None)
    assert system_stats.NvmlGpu(0).read() is None


def _nvml_present():
    try:
        return system_stats._load_nvml() is not None
    except OSError:
        return False


@pytest.mark.skipif(not _nvml_present(), reason="no NVIDIA driver (nvml) on this machine")
def test_real_nvml_reads_this_machines_card():
    stats = system_stats.NvmlGpu(0).read()
    assert stats is not None
    assert stats.total_mb > 1000 and 0 < stats.used_mb <= stats.total_mb
    assert system_stats.NvmlGpu(0).name()
    assert stats.util_percent is None or 0 <= stats.util_percent <= 100
    # struct layouts from nvml.h: 3 x unsigned long long, 2 x unsigned int
    assert ctypes.sizeof(system_stats._Memory) == 24
    assert ctypes.sizeof(system_stats._Utilization) == 8
