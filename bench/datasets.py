"""Test data: a fixed, seeded sample of public Hugging Face speech datasets.

The data comes straight from the dataset's parquet files on the Hub (pyarrow reads them
through huggingface_hub's HfFileSystem with HTTP range requests). Parquet stores rows in
groups of ~100 and a group is the smallest piece that can be read, so the sample is drawn
in two steps: seeded random row groups, then PER_GROUP seeded random rows inside each.
Only those groups are downloaded (~550 MB for all three datasets instead of the full
sets). The dataset revision is pinned in the manifest at the first download, so later
runs read the same files and draw exactly the same clips even after the dataset changes.

Rows are visited strictly in that seeded order. Content problems (undecodable audio,
length outside MIN_SEC..MAX_SEC, reference shorter than 2 characters) skip a row for
good; a network failure that survives the retries stops the download instead of
skipping, so a later run continues at the same row and the sample stays identical.

Layout: bench_clips/<key>/<row:06d>.wav (16 kHz mono PCM_16) + <row:06d>.txt (reference)
and bench_clips/<key>/manifest.json (row = position in the split, files in name order)."""
import io
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

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

DATASETS = {
    "anime": {"hf_id": "joujiboi/japanese-anime-speech-v2", "split": "sfw", "n": 200,
              "category": "anime", "label": "動畫（japanese-anime-speech-v2，sfw）"},
    "vtuber": {"hf_id": "japanese-asr/ja_asr.reazonspeech_test", "split": "test", "n": 200,
               "category": "vtuber", "label": "VTuber 近似（ReazonSpeech 電視語音）"},
    "baseline": {"hf_id": "japanese-asr/ja_asr.common_voice_8_0", "split": "test", "n": 100,
                 "category": "baseline", "label": "基準（Common Voice 8.0）"},
}
LIVE_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")
APPROX_TOTAL_MB = 550

LICENSE_NOTE = (
    "授權說明：動畫資料集（japanese-anime-speech-v2）標示 GPL，音源是有版權的動畫；"
    "ReazonSpeech 依日本著作權法第 30-4 條僅限資訊分析用途；Common Voice 為 CC0。"
    "下載的音檔與報告只留在本機（bench_clips/、bench_results/ 不會加入 git），請勿散布。"
)


class DownloadError(RuntimeError):
    """The download stopped (network); run it again to continue at the same row."""


@dataclass(frozen=True)
class Clip:
    dataset: str
    id: str
    wav: Path
    ref: str
    duration: float


# ---------------------------------------------------------------- Hub source


class HubParquetSource:
    """One dataset split on the Hugging Face Hub, read from its parquet files.

    `list_files(revision)` and `open_file(revision, path)` default to HfApi /
    HfFileSystem; tests pass local implementations. Every network call is retried with
    backoff and ends in DownloadError."""

    def __init__(self, hf_id: str, split: str, list_files=None, open_file=None, get_revision=None,
                 retries: int = 4, sleep=time.sleep):
        self.hf_id = hf_id
        self.split = split
        self._list_files = list_files or self._hub_list_files
        self._open_file = open_file or self._hub_open_file
        self._get_revision = get_revision or self._hub_revision
        self._retries = retries
        self._sleep = sleep
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
    def _retry(self, what, fn):
        delay = 2.0
        for attempt in range(self._retries + 1):
            try:
                return fn()
            except DownloadError:
                raise  # not a network problem: retrying cannot help
            except Exception as e:  # huggingface_hub / fsspec / httpx raise many types
                if attempt == self._retries:
                    raise DownloadError(f"{what}: {type(e).__name__}: {e}") from e
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
            raise DownloadError(f"{self.hf_id}: no parquet files for split '{self.split}'")
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
                    raise DownloadError(f"{self.hf_id}/{file}: unexpected columns {pf.schema_arrow.names}")
                return pf.read_row_group(group, columns=[audio_col, "transcription"]).to_pylist(), audio_col

        rows, audio_col = self._retry(f"{self.hf_id}/{file} group {group}", read)
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
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def manifest_path(key: str) -> Path:
    return paths.clips_dir() / key / "manifest.json"


def read_manifest(key: str):
    p = manifest_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning(f"Unreadable manifest {p}: {e}; starting the dataset again")
        return None


def dataset_status(key: str) -> tuple[int, int]:
    """(clips ready, clips wanted) for the menu header."""
    spec = DATASETS[key]
    m = read_manifest(key)
    if not m or m.get("method") != METHOD:
        return 0, spec["n"]
    return len(m.get("clips", [])), spec["n"]


def _check_clip(info: dict):
    """(audio, duration, ref) or a skip reason string."""
    ref = info["text"].strip()
    if not info.get("audio"):
        return "no audio"
    if len(normalize_ja(ref)) < 2:
        return "reference too short"
    try:
        audio = decode_to_16k(info["audio"])
    except ValueError as e:
        return str(e)[:120]
    duration = len(audio) / SAMPLE_RATE
    if not (MIN_SEC <= duration <= MAX_SEC):
        return f"duration {duration:.1f}s outside {MIN_SEC}-{MAX_SEC}s"
    return audio, duration, ref


def download_dataset(key: str, source, n: int | None = None, progress=None, info=None) -> dict:
    """Fetch (or continue fetching) a dataset sample. Returns the manifest; raises
    DownloadError on a network failure (progress so far is kept). `info(text)` receives
    one-off messages such as the download size."""
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
        if m:
            log.info(f"{key}: manifest from another sampling method or size; starting the sample again")
        m = {"method": METHOD, "hf_id": spec["hf_id"], "split": spec["split"], "seed": SEED, "n_target": n,
             "per_group": PER_GROUP, "filters": {"min_sec": MIN_SEC, "max_sec": MAX_SEC, "min_ref_chars": 2},
             "revision": None, "layout": None, "warmup": [], "clips": [], "skipped": [], "complete": False}
    if m.get("complete") and len(m["warmup"]) + len(m["clips"]) >= target:
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

    done_rows = {c["row"] for c in m["warmup"] + m["clips"]}
    skipped_rows = {s["row"] for s in m["skipped"]}
    for gi, offsets in sample_plan(key, layout):
        if len(m["warmup"]) + len(m["clips"]) >= target:
            break
        rows = [base[gi] + r for r in offsets]
        if all(r in done_rows or r in skipped_rows for r in rows):
            continue
        g = layout[gi]
        records = source.read_group(m["revision"], g["file"], g["group"])
        for offset, row in zip(offsets, rows):
            if len(m["warmup"]) + len(m["clips"]) >= target:
                break
            if row in done_rows or row in skipped_rows:
                continue
            checked = _check_clip(records[offset]) if offset < len(records) else "row missing in its group"
            if isinstance(checked, str):
                m["skipped"].append({"row": row, "reason": checked})
                skipped_rows.add(row)
                _atomic_write_json(mpath, m)
                continue
            audio, duration, ref = checked
            cid = f"{row:06d}"
            try:
                sf.write(str(out / f"{cid}.wav"), audio, SAMPLE_RATE, subtype="PCM_16")
                (out / f"{cid}.txt").write_text(ref, encoding="utf-8")
            except (OSError, RuntimeError) as e:  # disk full / no permission: stop, keep the progress
                raise DownloadError(f"cannot write {out / cid}: {e}") from e
            entry = {"id": cid, "row": row, "duration": round(duration, 3), "ref": ref}
            (m["warmup"] if len(m["warmup"]) < WARMUP_CLIPS else m["clips"]).append(entry)
            done_rows.add(row)
            _atomic_write_json(mpath, m)
            if progress:
                progress(key, len(m["clips"]), n)
    m["complete"] = len(m["warmup"]) + len(m["clips"]) >= target
    _atomic_write_json(mpath, m)
    if not m["complete"]:
        log.warning(f"{key}: only {len(m['clips'])}/{n} usable clips in the whole split")
    return m


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
