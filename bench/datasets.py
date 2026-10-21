"""Test data: a fixed, seeded sample of public Hugging Face speech datasets.

The data comes straight from the dataset's parquet files on the Hub (pyarrow reads them
through huggingface_hub's HfFileSystem with HTTP range requests). A parquet file stores
rows in row groups (the Hub's audio datasets use ~100 rows each; the code reads the real
sizes from the file footers) and a group is the smallest piece that can be read, so the
sample is drawn in two steps: seeded random row groups, then PER_GROUP seeded random rows
inside each. Only those groups are downloaded (~550 MB for all three datasets instead of
the full sets). The dataset revision is pinned in the manifest at the first download, so
later runs read the same files and draw exactly the same clips even after the dataset
changes (if the pinned revision disappears from the Hub, delete manifest.json to start
over). The first WARMUP_CLIPS usable rows warm the models up and are not scored.

Rows are visited strictly in that seeded order. Rows whose data is unusable (no audio,
length outside MIN_SEC..MAX_SEC, reference shorter than 2 characters after
normalize_ja, row missing in its group) are skipped for good. Audio that cannot be
decoded is skipped for this download only, since a missing codec on this computer looks
the same. When most rows are skipped the download stops (MAX_SKIP_SHARE): that is a
problem of the data or of this computer, not bad luck. A network failure that survives
the retries stops the download instead of skipping, so a later run continues at the
same row and the sample stays identical.

Layout: bench_clips/<key>/<row:06d>.wav (16 kHz mono PCM_16) + <row:06d>.txt (reference)
and bench_clips/<key>/manifest.json (row = position in the split, files in name order)."""
import hashlib
import io
import json
import logging
import random
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from bench import paths
from bench.textnorm import normalize_ja

log = logging.getLogger("LiveTranslate.Bench.Data")

SEED = 20261020
METHOD = "hub-parquet-v1"
PER_GROUP = 8
WARMUP_CLIPS = 3
MIN_SEC, MAX_SEC = 0.8, 12.0
SAMPLE_RATE = 16000
SKIP_MARGIN = 1.15  # expected share of rows skipped for content reasons, for size estimates
MAX_SKIP_SHARE = 0.5  # more rows skipped than this (after SKIP_CHECK_ROWS rows): stop
SKIP_CHECK_ROWS = 40
CALL_TIMEOUT_S = 120.0  # one Hub request: revision, file list, footer
GROUP_TIMEOUT_S = 600.0  # one row group (tens of MB in 1 MB range reads)

# The keys double as the analysis categories (anime, vtuber); baseline is not ranked
DATASETS = {
    "anime": {"hf_id": "joujiboi/japanese-anime-speech-v2", "split": "sfw", "n": 200,
              "label": "動畫（japanese-anime-speech-v2，sfw）"},
    "vtuber": {"hf_id": "japanese-asr/ja_asr.reazonspeech_test", "split": "test", "n": 200,
               "label": "VTuber 近似（ReazonSpeech 電視語音）"},
    "baseline": {"hf_id": "japanese-asr/ja_asr.common_voice_8_0", "split": "test", "n": 100,
                 "label": "基準（Common Voice 8.0）"},
}
LIVE_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")
APPROX_TOTAL_MB = 550  # measured once; also in launcher/bench.ps1 and CLAUDE.md (update with PER_GROUP / n)

LICENSE_NOTE = (
    "授權說明：動畫資料集（japanese-anime-speech-v2）標示 GPL，音源是有版權的動畫；"
    "ReazonSpeech 依日本著作權法第 30-4 條僅限資訊分析用途；Common Voice 為 CC0。"
    "下載的音檔與報告只留在本機（bench_clips/、bench_results/ 不會加入 git），請勿散布。"
)


class DownloadError(RuntimeError):
    """The download stopped; the progress so far is kept."""


class NetworkError(DownloadError):
    """Network trouble that survived the retries; a rerun continues at the same row."""


class DataError(DownloadError):
    """The dataset is not what this code expects (files missing or renamed, other
    columns, access refused, most rows unusable) or huggingface_hub changed: retrying
    cannot help."""


class DiskError(DownloadError):
    """A clip or the manifest could not be written (disk full, no permission)."""


def _transient(e: BaseException) -> bool:
    """Worth retrying: timeouts, dropped connections, rate limits and server errors."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    if isinstance(e, (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError)):
        return False
    if isinstance(e, OSError):  # includes TimeoutError, ConnectionError and pyarrow's IO errors
        return True
    # httpx / httpx2 transport errors (connect / read / protocol) do not derive from OSError
    return any(c.__name__ in ("TransportError", "TimeoutException") for c in type(e).__mro__)


def _call_with_timeout(fn, timeout: float):
    """fn() on a daemon thread; TimeoutError when it has not returned in `timeout` s.
    huggingface_hub 2.x lists files and resolves paths without any HTTP timeout, so a
    silently dropped connection would otherwise block forever. A call that times out
    keeps its thread until the connection dies; the retry uses a new one."""
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:  # handed over to the caller below
            box["error"] = e

    t = threading.Thread(target=run, name="bench-hub", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"no answer from the Hub in {timeout:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


@dataclass(frozen=True)
class Clip:
    dataset: str
    id: str
    wav: Path
    ref: str
    duration: float

    @property
    def key(self) -> str:
        """The item key of this clip in result records ("<dataset>/<id>")."""
        return f"{self.dataset}/{self.id}"


# ---------------------------------------------------------------- Hub source


class HubParquetSource:
    """One dataset split on the Hugging Face Hub, read from its parquet files.

    `list_files(revision)` and `open_file(revision, path)` default to HfApi /
    HfFileSystem; tests pass local implementations. Each step (revision, file list, one
    footer, one row group) runs under a time limit (`call_timeout`, `group_timeout`).
    Network trouble is retried `retries` times (backoff 2 s, doubling) and then raises
    NetworkError; anything else raises DataError at once."""

    def __init__(self, hf_id: str, split: str, list_files=None, open_file=None, get_revision=None,
                 retries: int = 3, sleep=time.sleep, call_timeout: float = CALL_TIMEOUT_S,
                 group_timeout: float = GROUP_TIMEOUT_S):
        if retries < 0:
            raise ValueError(f"retries must be >= 0, not {retries}")
        self.hf_id = hf_id
        self.split = split
        self._list_files = list_files or self._hub_list_files
        self._open_file = open_file or self._hub_open_file
        self._get_revision = get_revision or self._hub_revision
        self._retries = retries
        self._sleep = sleep
        self._call_timeout = call_timeout
        self._group_timeout = group_timeout
        self._fs = None

    # ------------------------------------------------------------ Hub defaults
    def _hub_revision(self):
        from huggingface_hub import HfApi

        return HfApi().dataset_info(self.hf_id, timeout=30).sha

    def _hub_list_files(self, revision):
        from huggingface_hub import HfApi

        return HfApi().list_repo_files(self.hf_id, repo_type="dataset", revision=revision)

    def _hub_open_file(self, revision, path):
        from huggingface_hub import HfFileSystem

        if self._fs is None:
            self._fs = HfFileSystem()
        return self._fs.open(f"datasets/{self.hf_id}@{revision}/{path}", "rb", block_size=1 << 20)

    # ------------------------------------------------------------ retries
    def _retry(self, what, fn, timeout=None):
        delay = 2.0
        for attempt in range(self._retries + 1):
            try:
                return _call_with_timeout(fn, timeout or self._call_timeout)
            except DownloadError:
                raise  # raised on purpose with its own reason
            except Exception as e:  # huggingface_hub / fsspec / httpx / pyarrow raise many types
                reason = f"{what}: {type(e).__name__}: {e}"
                if not _transient(e):
                    raise DataError(reason) from e
                if attempt == self._retries:
                    raise NetworkError(reason) from e
                log.warning(f"{what} failed ({type(e).__name__}: {e}), retrying in {delay:.0f}s")
                self._sleep(delay)
                delay = min(delay * 2, 60.0)

    # ------------------------------------------------------------ API
    def revision(self) -> str:
        return self._retry(f"{self.hf_id}: dataset revision", self._get_revision)

    def layout(self, revision) -> list:
        """[{"file", "group", "rows", "bytes"}] for every row group of the split's parquet
        files (name order). Only the file footers are read."""
        import pyarrow.parquet as pq

        files = self._retry(f"{self.hf_id}: file list", lambda: self._list_files(revision))
        names = sorted(f for f in files if f.endswith(".parquet") and f.split("/")[-1].startswith(self.split + "-"))
        if not names:
            raise DataError(f"{self.hf_id}: no parquet files for split '{self.split}'")
        groups = []
        for name in names:
            def footer(name=name):
                with self._open_file(revision, name) as fh:
                    return pq.ParquetFile(fh).metadata
            meta = self._retry(f"{self.hf_id}/{name}: footer", footer)
            for g in range(meta.num_row_groups):
                rg = meta.row_group(g)
                groups.append({"file": name, "group": g, "rows": rg.num_rows, "bytes": rg.total_byte_size})
        return groups

    def read_group(self, revision, file: str, group: int) -> list:
        """[{"audio": bytes or None, "text": str}] for one row group."""
        import pyarrow.parquet as pq

        def read():
            with self._open_file(revision, file) as fh:
                pf = pq.ParquetFile(fh)
                audio_col = next((f.name for f in pf.schema_arrow
                                  if hasattr(f.type, "get_field_index") and f.type.get_field_index("bytes") >= 0), None)
                if audio_col is None or "transcription" not in pf.schema_arrow.names:
                    raise DataError(f"{self.hf_id}/{file}: unexpected columns {pf.schema_arrow.names}")
                return pf.read_row_group(group, columns=[audio_col, "transcription"]).to_pylist(), audio_col

        rows, audio_col = self._retry(f"{self.hf_id}/{file} group {group}", read, timeout=self._group_timeout)
        return [{"audio": (r.get(audio_col) or {}).get("bytes"), "text": str(r.get("transcription") or "")}
                for r in rows]


def make_source(key: str) -> HubParquetSource:
    spec = DATASETS[key]
    return HubParquetSource(spec["hf_id"], spec["split"])


# ---------------------------------------------------------------- sampling and audio


def sample_plan(key: str, layout: list, per_group: int = PER_GROUP):
    """Seeded (group index, [row offsets]) pairs, groups in random order. A string seed
    is hashed with sha512 by random.Random, so PYTHONHASHSEED does not matter. The draws
    happen in a fixed sequence, so a resumed download sees the same plan."""
    rng = random.Random(f"{SEED}:{key}:{METHOD}")
    order = list(range(len(layout)))
    rng.shuffle(order)
    for gi in order:
        rows = layout[gi]["rows"]
        yield gi, sorted(rng.sample(range(rows), min(per_group, rows)))


def estimate_mb(layout: list, n: int, per_group: int = PER_GROUP) -> float:
    if not layout:
        return 0.0
    groups = min(len(layout), int(np.ceil((n + WARMUP_CLIPS) * SKIP_MARGIN / per_group)))
    avg = sum(g["bytes"] for g in layout) / len(layout)
    return groups * avg / 1e6


def decode_to_16k(data: bytes) -> np.ndarray:
    """Any soundfile-readable audio (wav / flac / mp3 / ogg) -> float32 mono 16 kHz."""
    import soundfile as sf

    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:  # soundfile raises its own LibsndfileError / RuntimeError
        raise ValueError(f"undecodable audio: {e}") from e
    mono = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        try:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="soxr_hq")
        except Exception as e:  # librosa / soxr errors: a content problem of this clip
            raise ValueError(f"cannot resample {sr} Hz audio: {e}") from e
    return np.asarray(mono, dtype=np.float32)


def load_audio_file(path) -> np.ndarray:
    return decode_to_16k(Path(path).read_bytes())


# ---------------------------------------------------------------- manifest


def _atomic_write_json(path: Path, data):
    from bench.state import atomic_write_json

    try:
        atomic_write_json(path, data)
    except OSError as e:
        raise DiskError(f"cannot write {path}: {e}") from e


def manifest_path(key: str) -> Path:
    return paths.clips_dir() / key / "manifest.json"


def read_manifest(key: str):
    p = manifest_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning(f"Unreadable manifest {p}: {e}")
        return None


def dataset_status(key: str) -> tuple[int, int]:
    """(clips ready, clips wanted) for the menu header."""
    spec = DATASETS[key]
    m = read_manifest(key)
    if not m or m.get("method") != METHOD:
        return 0, spec["n"]
    return len(m.get("clips", [])), spec["n"]


class Skip(NamedTuple):
    reason: str
    lasting: bool  # a property of the data (saved in the manifest), not of this computer


def _check_clip(info: dict):
    """(audio, duration, ref) or a Skip."""
    ref = info["text"].strip()
    if not info.get("audio"):
        return Skip("no audio", True)
    if len(normalize_ja(ref)) < 2:
        return Skip("reference too short", True)
    try:
        audio = decode_to_16k(info["audio"])
    except ValueError as e:  # a broken file, or a codec missing on this computer
        return Skip(str(e)[:120], False)
    duration = len(audio) / SAMPLE_RATE
    if not (MIN_SEC <= duration <= MAX_SEC):
        return Skip(f"duration {duration:.1f}s outside {MIN_SEC}-{MAX_SEC}s", True)
    return audio, duration, ref


def _collected(m: dict) -> int:
    return len(m["warmup"]) + len(m["clips"])


def _check_skip_share(key: str, checked: int, skips: Counter):
    """Stop when most rows of this download were unusable: a broken dataset or a missing
    codec, which would otherwise read the whole split (far more than the estimate)."""
    skipped = sum(skips.values())
    if checked >= SKIP_CHECK_ROWS and skipped > checked * MAX_SKIP_SHARE:
        top = "；".join(f"{reason} ×{count}" for reason, count in skips.most_common(3))
        raise DataError(f"{key}: {skipped} of {checked} rows unusable ({top})")


def download_dataset(key: str, source, n: int | None = None, progress=None, info=None) -> dict:
    """Fetch (or continue fetching) a dataset sample. Returns the manifest; raises a
    DownloadError subclass when it stops (progress so far is kept). `progress(key, clips,
    n, skipped)` follows every row; `info(text)` receives one-off messages such as the
    download size."""
    import soundfile as sf

    spec = DATASETS[key]
    n = spec["n"] if n is None else n
    target = n + WARMUP_CLIPS
    out = paths.clips_dir() / key
    out.mkdir(parents=True, exist_ok=True)
    mpath = manifest_path(key)

    m = read_manifest(key)
    same = m and m.get("method") == METHOD and m.get("hf_id") == spec["hf_id"] and m.get("split") == spec["split"] \
        and m.get("seed") == SEED and m.get("n_target") == n and m.get("per_group") == PER_GROUP
    if not same:
        if m or mpath.exists():
            log.warning(f"{key}: manifest unreadable or from another sampling method or size; "
                        "drawing the sample again (earlier benchmark runs used other clips)")
        m = {"method": METHOD, "hf_id": spec["hf_id"], "split": spec["split"], "seed": SEED, "n_target": n,
             "per_group": PER_GROUP, "filters": {"min_sec": MIN_SEC, "max_sec": MAX_SEC, "min_ref_chars": 2},
             "revision": None, "layout": None, "warmup": [], "clips": [], "skipped": [], "complete": False}
    if m.get("complete") and _collected(m) >= target:
        return m

    if not m.get("revision"):
        m["revision"] = source.revision()
        _atomic_write_json(mpath, m)
    if not m.get("layout"):
        m["layout"] = source.layout(m["revision"])
        _atomic_write_json(mpath, m)
    layout = m["layout"]
    if info:
        info(f"預計下載約 {estimate_mb(layout, n):.0f} MB")
    base = []
    total = 0
    for g in layout:
        base.append(total)
        total += g["rows"]

    seen = {c["row"] for c in m["warmup"] + m["clips"]} | {s["row"] for s in m["skipped"]}
    checked_now = 0
    skips_now = Counter()  # this download only: the brake judges what is happening now
    for gi, offsets in sample_plan(key, layout):
        if _collected(m) >= target:
            break
        rows = [base[gi] + r for r in offsets]
        if all(r in seen for r in rows):
            continue
        g = layout[gi]
        records = source.read_group(m["revision"], g["file"], g["group"])
        for offset, row in zip(offsets, rows):
            if _collected(m) >= target:
                break
            if row in seen:
                continue
            seen.add(row)
            checked_now += 1
            checked = _check_clip(records[offset]) if offset < len(records) else Skip("row missing in its group", True)
            if isinstance(checked, Skip):
                skips_now[re.sub(r"\d+(\.\d+)?", "#", checked.reason.split(":")[0])] += 1
                if checked.lasting:
                    m["skipped"].append({"row": row, "reason": checked.reason})
                    _atomic_write_json(mpath, m)
                else:
                    log.warning(f"{key}: row {row} skipped for this download: {checked.reason}")
                if progress:
                    progress(key, len(m["clips"]), n, sum(skips_now.values()))
                _check_skip_share(key, checked_now, skips_now)
                continue
            audio, duration, ref = checked
            cid = f"{row:06d}"
            try:
                sf.write(str(out / f"{cid}.wav"), audio, SAMPLE_RATE, subtype="PCM_16")
                (out / f"{cid}.txt").write_text(ref, encoding="utf-8")
            except (OSError, RuntimeError) as e:  # disk full / no permission: stop, keep the progress
                raise DiskError(f"cannot write {out / cid}: {e}") from e
            entry = {"id": cid, "row": row, "duration": round(duration, 3), "ref": ref}
            (m["warmup"] if len(m["warmup"]) < WARMUP_CLIPS else m["clips"]).append(entry)
            _atomic_write_json(mpath, m)
            if progress:
                progress(key, len(m["clips"]), n, sum(skips_now.values()))
    m["complete"] = _collected(m) >= target
    _atomic_write_json(mpath, m)
    if not m["complete"]:
        log.warning(f"{key}: only {len(m['clips'])}/{n} usable clips in the whole split")
    return m


def sample_id(key: str, limit: int | None = None) -> dict | None:
    """What identifies the test clips of a dataset (sampling method, pinned revision and
    the first `limit` clip ids), so a later run can tell whether it measures the same
    sentences. None without a usable manifest."""
    m = read_manifest(key)
    if not m or m.get("method") != METHOD:
        return None
    ids = [c["id"] for c in m.get("clips", [])]
    ids = ids[:limit] if limit else ids
    return {"method": m["method"], "revision": m.get("revision"),
            "clips": hashlib.sha1(",".join(ids).encode("utf-8")).hexdigest()}


def load_clips(key: str, limit: int | None = None) -> tuple[list[Clip], list[Clip]]:
    """(warm-up clips, test clips) whose files exist."""
    m = read_manifest(key)
    if not m or m.get("method") != METHOD:
        return [], []
    base = paths.clips_dir() / key

    def build(entries):
        out = []
        for e in entries:
            wav = base / f"{e['id']}.wav"
            if wav.exists():
                out.append(Clip(key, e["id"], wav, e["ref"], float(e["duration"])))
            else:
                log.warning(f"{key}: missing {wav.name}, skipped")
        return out

    clips = build(m.get("clips", []))
    return build(m.get("warmup", [])), (clips[:limit] if limit else clips)


def live_recordings() -> list[tuple[Path, str]]:
    """User recordings for the VAD stage: bench_clips/live/* (VTuber) and
    bench_clips/live/anime/* (anime)."""
    found = []
    root = paths.live_dir()
    for folder, category in ((root, "vtuber"), (root / "anime", "anime")):
        if folder.is_dir():
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.suffix.lower() in LIVE_EXTENSIONS:
                    found.append((f, category))
    return found
