"""Progress line and time-remaining estimate."""
import sys
import time


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} 小時 {minutes} 分"
    if minutes:
        return f"{minutes} 分"
    return f"{secs} 秒"


class Ewma:
    def __init__(self, alpha: float = 0.2, initial: float | None = None):
        self.alpha = alpha
        self.value = initial

    def add(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class ProgressLine:
    """Rewrites one console line at most once per `min_interval` seconds."""

    def __init__(self, out=None, min_interval: float = 1.0, clock=time.monotonic):
        self._out = out or sys.stdout
        self._min_interval = min_interval
        self._clock = clock
        self._last = None
        self._width = 0
        self.started = clock()

    def show(self, stage: str, combo: str, done: int, total: int, avg_ms: float | None, remaining_s: float | None,
             force: bool = False):
        now = self._clock()
        if not force and self._last is not None and now - self._last < self._min_interval:
            return
        self._last = now
        parts = [f"[{stage}]", combo, f"第 {done}/{total} 句"]
        if avg_ms is not None:
            parts.append(f"平均 {avg_ms:.0f} ms")
        parts.append(f"已用 {fmt_duration(now - self.started)}")
        if remaining_s is not None:
            parts.append(f"預估剩餘 {fmt_duration(remaining_s)}")
        text = "｜".join(parts)
        pad = max(0, self._width - len(text))
        self._width = len(text)
        self._out.write("\r" + text + " " * pad)
        self._out.flush()

    def finish(self, message: str = ""):
        self._out.write("\r" + message + " " * max(0, self._width - len(message)) + "\n")
        self._out.flush()
        self._width = 0
        self._last = None
