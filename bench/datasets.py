"""Test data: a fixed, seeded sample of public Hugging Face speech datasets.

Only the sampled rows are fetched, through the datasets-server API (no full download,
no `datasets` package): /size gives the row count, /rows?offset=N&length=1 one row with
a signed audio URL and its transcription. Candidates are visited strictly in seeded
order, so every run gets the same clips. Content problems (undecodable audio, length
outside MIN_SEC..MAX_SEC, reference shorter than 2 characters) skip a row for good; a
network failure that survives the retries stops the download instead of skipping, so a
later resume continues at the same row and the sample stays identical.

Layout: bench_clips/<key>/<row:06d>.wav (16 kHz mono PCM_16) + <row:06d>.txt (reference)
and bench_clips/<key>/manifest.json."""
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
WARMUP_CLIPS = 3
MIN_SEC, MAX_SEC = 0.8, 12.0
SAMPLE_RATE = 16000
ROWS_API = "https://datasets-server.huggingface.co"
HUB_API = "https://huggingface.co/api/datasets"

DATASETS = {
    "anime": {"hf_id": "joujiboi/japanese-anime-speech-v2", "split": "sfw", "n": 200,
              "category": "anime", "label": "動畫（japanese-anime-speech-v2，sfw）"},
    "vtuber": {"hf_id": "japanese-asr/ja_asr.reazonspeech_test", "split": "test", "n": 200,
               "category": "vtuber", "label": "VTuber 近似（ReazonSpeech 電視語音）"},
    "baseline": {"hf_id": "japanese-asr/ja_asr.common_voice_8_0", "split": "test", "n": 100,
                 "category": "baseline", "label": "基準（Common Voice 8.0）"},
}
LIVE_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")

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


# ---------------------------------------------------------------- HTTP


class RowsClient:
    """datasets-server / Hub API calls with retries. `http` is an httpx.Client
    (tests pass one with a MockTransport); `sleep` is injectable."""

    def __init__(self, http, sleep=time.sleep, min_interval=0.25, retries=5):
        self._http = http
        self._sleep = sleep
        self._min_interval = min_interval
        self._retries = retries
        self._last = 0.0

    def _get(self, url, params=None):
        import httpx

        delay = 2.0
        last_error = None
        for attempt in range(self._retries + 1):
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self._http.get(url, params=params)
            except httpx.TransportError as e:  # includes timeouts
                last_error = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}"
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = max(delay, float(retry_after))
                else:
                    raise DownloadError(f"HTTP {resp.status_code} for {url}: {resp.text[:200]}")
            if attempt < self._retries:
                log.warning(f"Request failed ({last_error}), retrying in {delay:.0f}s: {url}")
                self._sleep(delay)
                delay = min(delay * 2, 32.0)
        raise DownloadError(f"{last_error} after {self._retries + 1} attempts: {url}")

    def _json(self, url, params=None):
        resp = self._get(url, params)
        try:
            return resp.json()
        except ValueError as e:
            raise DownloadError(f"not JSON from {url}: {resp.text[:120]!r}") from e

    def num_rows(self, hf_id, split) -> int:
        data = self._json(f"{ROWS_API}/size", {"dataset": hf_id})
        for s in (data.get("size") or {}).get("splits") or []:
            if s.get("split") == split and isinstance(s.get("num_rows"), int):
                return s["num_rows"]
        raise DownloadError(f"{hf_id}: split '{split}' not found in /size")

    def revision(self, hf_id):
        try:
            return self._json(f"{HUB_API}/{hf_id}").get("sha")
        except (DownloadError, AttributeError) as e:
            log.warning(f"Could not read the dataset revision of {hf_id}: {e}")
            return None

    def row(self, hf_id, split, offset) -> dict:
        data = self._json(f"{ROWS_API}/rows", {"dataset": hf_id, "config": "default", "split": split,
                                               "offset": offset, "length": 1})
        rows = data.get("rows") or []
        if not rows:
            raise DownloadError(f"{hf_id}: no row at offset {offset}")
        row = rows[0].get("row") or {}
        audio = row.get("audio")
        if not (isinstance(audio, list) and audio and isinstance(audio[0], dict) and audio[0].get("src")):
            return {"audio_url": None, "text": str(row.get("transcription") or "")}
        return {"audio_url": audio[0]["src"], "text": str(row.get("transcription") or "")}

    def fetch(self, url) -> bytes:
        return self._get(url).content


def make_http_client(proxy: str = "system"):
    import httpx

    headers = {"User-Agent": "LiveTranslate-bench"}
    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception as e:  # token lookup is optional
        log.debug(f"No Hugging Face token: {e}")
        token = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    timeout = httpx.Timeout(30.0, connect=10.0)
    if proxy in ("none", "", None):
        return httpx.Client(timeout=timeout, headers=headers, trust_env=False, follow_redirects=True)
    if proxy == "system":
        return httpx.Client(timeout=timeout, headers=headers, follow_redirects=True)
    return httpx.Client(timeout=timeout, headers=headers, proxy=proxy, follow_redirects=True)


# ---------------------------------------------------------------- sampling and audio


def sample_candidates(key: str, num_rows: int, needed: int) -> list[int]:
    """Seeded candidate rows for a dataset; a string seed is hashed with sha512 by
    random.Random, so PYTHONHASHSEED does not matter."""
    rng = random.Random(f"{SEED}:{key}")
    return rng.sample(range(num_rows), min(num_rows, needed * 3))


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
    return (len(m.get("clips", [])) if m else 0), spec["n"]


def download_dataset(key: str, client: RowsClient, n: int | None = None, progress=None) -> dict:
    """Fetch (or continue fetching) a dataset sample. Returns the manifest; raises
    DownloadError on a network failure (progress so far is kept)."""
    spec = DATASETS[key]
    n = spec["n"] if n is None else n
    target = n + WARMUP_CLIPS
    out = paths.clips_dir() / key
    out.mkdir(parents=True, exist_ok=True)
    mpath = manifest_path(key)

    m = read_manifest(key)
    same = m and m.get("hf_id") == spec["hf_id"] and m.get("split") == spec["split"] \
        and m.get("seed") == SEED and m.get("n_target") == n
    if not same:
        m = {"hf_id": spec["hf_id"], "split": spec["split"], "seed": SEED, "n_target": n,
             "filters": {"min_sec": MIN_SEC, "max_sec": MAX_SEC, "min_ref_chars": 2},
             "num_rows": None, "revision_sha": None, "warmup": [], "clips": [], "skipped": [],
             "complete": False}
    if m.get("complete") and len(m["warmup"]) + len(m["clips"]) >= target:
        return m

    num_rows = client.num_rows(spec["hf_id"], spec["split"])
    if m.get("num_rows") not in (None, num_rows):
        log.warning(f"{key}: row count changed {m['num_rows']} -> {num_rows}; the sample may differ")
    m["num_rows"] = num_rows
    sha = client.revision(spec["hf_id"])
    if m.get("revision_sha") and sha and sha != m["revision_sha"]:
        log.warning(f"{key}: dataset updated ({m['revision_sha'][:8]} -> {sha[:8]}); keeping the clips already saved")
    m["revision_sha"] = m.get("revision_sha") or sha

    done_rows = {c["row"] for c in m["warmup"] + m["clips"]}
    skipped_rows = {s["row"] for s in m["skipped"]}
    for row in sample_candidates(key, num_rows, target):
        if len(m["warmup"]) + len(m["clips"]) >= target:
            break
        if row in done_rows or row in skipped_rows:
            continue
        info = client.row(spec["hf_id"], spec["split"], row)
        reason = None
        ref = info["text"].strip()
        if not info["audio_url"]:
            reason = "no audio"
        elif len(normalize_ja(ref)) < 2:
            reason = "reference too short"
        else:
            data = client.fetch(info["audio_url"])  # signed URL: use it right away
            try:
                audio = decode_to_16k(data)
            except ValueError as e:
                reason = str(e)[:120]
            else:
                duration = len(audio) / SAMPLE_RATE
                if not (MIN_SEC <= duration <= MAX_SEC):
                    reason = f"duration {duration:.1f}s outside {MIN_SEC}-{MAX_SEC}s"
        if reason:
            m["skipped"].append({"row": row, "reason": reason})
            _atomic_write_json(mpath, m)
            continue
        import soundfile as sf

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
        log.warning(f"{key}: only {len(m['clips'])}/{n} usable clips among the sampled candidates")
    return m


def load_clips(key: str, limit: int | None = None) -> tuple[list[Clip], list[Clip]]:
    """(warm-up clips, test clips) whose files exist."""
    m = read_manifest(key)
    if not m:
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
