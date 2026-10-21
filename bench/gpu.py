"""Whole-card VRAM: peak sampling while a model runs, and waiting until memory falls
back after an unload. NVML (system_stats.NvmlGpu) only reports the whole card on
Windows, so other programs add noise; the run measures a baseline first."""
import logging
import threading
import time

log = logging.getLogger("LiveTranslate.Bench.GPU")


class VramReader:
    def __init__(self, index: int = 0, gpu=None):
        if gpu is None:
            import system_stats

            gpu = system_stats.NvmlGpu(index)
        self._gpu = gpu

    def used_mb(self):
        stats = self._gpu.read()
        return None if stats is None else float(stats.used_mb)


class VramSampler:
    """Background sampling of used VRAM; peak_mb is None when NVML is unavailable."""

    def __init__(self, reader, interval: float = 0.1):
        self._reader = reader
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.peak_mb = None

    def _sample(self):
        used = self._reader.used_mb()
        if used is not None and (self.peak_mb is None or used > self.peak_mb):
            self.peak_mb = used

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self._interval)

    def __enter__(self):
        self._sample()
        self._thread = threading.Thread(target=self._run, name="bench-vram", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._sample()
        return False


def wait_vram_settle(reader, baseline_mb, timeout=90.0, tolerance_mb=300.0, poll=0.5,
                     stable_after=15.0, stable_window=3.0, stable_delta_mb=50.0, high_limit_mb=1024.0,
                     clock=time.monotonic, sleep=time.sleep) -> dict:
    """Wait until used VRAM is back near the idle baseline.

    status: "settled" (<= baseline + tolerance), "stable_high" (after `stable_after` s
    it stopped moving within baseline + high_limit), "timeout", or "no_nvml" (fixed 10 s
    wait instead)."""
    start = clock()
    used = reader.used_mb()
    if baseline_mb is None or used is None:
        sleep(10.0)
        return {"status": "no_nvml", "used_mb": None, "waited_s": round(clock() - start, 1)}
    history = []
    while True:
        now = clock()
        history.append((now, used))
        if used <= baseline_mb + tolerance_mb:
            return {"status": "settled", "used_mb": used, "waited_s": round(now - start, 1)}
        recent = [u for t, u in history if now - t <= stable_window]
        if (now - start >= stable_after and len(recent) >= 2 and max(recent) - min(recent) < stable_delta_mb
                and used <= baseline_mb + high_limit_mb):
            log.warning(f"VRAM stable but {used - baseline_mb:.0f} MB above baseline")
            return {"status": "stable_high", "used_mb": used, "waited_s": round(now - start, 1)}
        if now - start >= timeout:
            log.warning(f"VRAM did not fall back within {timeout:.0f}s ({used - baseline_mb:.0f} MB above baseline)")
            return {"status": "timeout", "used_mb": used, "waited_s": round(now - start, 1)}
        sleep(poll)
        used = reader.used_mb()
        if used is None:
            return {"status": "no_nvml", "used_mb": None, "waited_s": round(clock() - start, 1)}
