"""Stage 2 VAD: run a recording through the app's VADProcessor in 32 ms chunks (as fast
as possible: segmentation depends only on the chunk sequence, not on wall time), and cut
the same recording into 20-30 s windows for the continuous reference transcript."""
import numpy as np

SR = 16000
CHUNK = 512  # 32 ms, Silero's native window
THRESHOLDS = (0.3, 0.4, 0.5, 0.6)
DEFAULT_THRESHOLD = 0.5
WINDOWS_PER_MINUTE = 2.5  # continuous windows are 20-30 s long


def estimate_windows(minutes: float, n_asr: int) -> float:
    """Judge calls of stage 2: every window, once per threshold, for every ASR."""
    return minutes * WINDOWS_PER_MINUTE * len(THRESHOLDS) * n_asr
VAD_SETTING_KEYS = ("vad_mode", "silence_mode", "silence_duration", "vad_min_speech_density",
                    "min_speech_duration", "max_speech_duration", "energy_threshold")


def make_vad(threshold: float, settings: dict):
    """A fresh VADProcessor per (recording, threshold): Silero keeps state and the class
    has no public reset."""
    from vad_processor import VADProcessor

    vad = VADProcessor(SR, threshold=threshold,
                       min_speech_duration=settings.get("min_speech_duration", 1.0),
                       max_speech_duration=settings.get("max_speech_duration", 15.0),
                       chunk_duration=CHUNK / SR)
    vad.update_settings({**{k: settings[k] for k in VAD_SETTING_KEYS if k in settings},
                         "vad_threshold": threshold})
    return vad


def segment_recording(audio: np.ndarray, vad) -> list:
    """[{"start", "end", "audio"}] in seconds. A segment ends where the samples still
    buffered in the VAD begin (backtrack splits keep a remainder); at the end silence is
    fed until the pause is long enough to close a segment (main.py's capture loop feeds
    silence when no audio arrives), then the VAD is flushed."""
    segs = []
    fed = 0

    def take(seg, end):
        if seg is not None and len(seg):
            segs.append({"start": max(0.0, (end - len(seg)) / SR), "end": end / SR, "audio": seg})

    for i in range(0, len(audio) - CHUNK + 1, CHUNK):
        fed += CHUNK
        seg = vad.process_chunk(np.asarray(audio[i:i + CHUNK], dtype=np.float32))
        take(seg, fed - vad._speech_samples)  # read-only peek at what stays buffered
    silence = np.zeros(CHUNK, dtype=np.float32)
    for _ in range(vad._get_effective_silence_limit() + 1):
        fed += CHUNK
        seg = vad.process_chunk(silence)
        take(seg, fed - vad._speech_samples)
    take(vad.flush(), fed)
    return segs


def continuous_windows(audio: np.ndarray, target_s: float = 30.0, search=(20.0, 30.0)) -> list:
    """[(start_sample, end_sample)]: windows of at most `target_s`, each cut at the
    quietest 32 ms between search[0] and search[1] seconds into it."""
    n = len(audio)
    windows = []
    pos = 0
    while pos < n:
        if n - pos <= int(target_s * SR):
            windows.append((pos, n))
            break
        lo = pos + int(search[0] * SR)
        hi = min(n - CHUNK, pos + int(search[1] * SR))
        starts = range(lo, hi, CHUNK)
        cut = min(starts, key=lambda s: float(np.mean(np.square(audio[s:s + CHUNK]))))
        windows.append((pos, cut))
        pos = cut
    return windows


def lines_in_window(segments: list, start_s: float, end_s: float) -> list:
    """Segments whose middle falls inside the window."""
    return [s for s in segments if start_s <= (s["start"] + s["end"]) / 2 < end_s]


def best_threshold(results: dict) -> float:
    """results: threshold -> errors per minute (hallucinated + missed). Lowest wins;
    ties go to the threshold closest to the default 0.5."""
    if not results:
        return DEFAULT_THRESHOLD
    return min(results, key=lambda th: (results[th], abs(th - DEFAULT_THRESHOLD)))
