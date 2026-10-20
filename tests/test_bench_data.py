"""bench.textnorm and bench.datasets: Japanese CER and the seeded sample read from the
datasets' parquet files. No network: real parquet files in a temp folder stand in for the
Hugging Face Hub (same reading code, local file opener)."""
import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from bench import datasets as ds
from bench.textnorm import corpus_cer, edit_distance, normalize_ja


# ---------------------------------------------------------------- textnorm

@pytest.mark.parametrize("raw, expected", [
    ("ＡＢＣ１２３", "abc123"),            # full-width ASCII -> half-width, lower case
    ("ｱﾘｶﾞﾄｳ", "アリガトウ"),             # half-width katakana -> full-width
    ("えーと、それは…。", "えーとそれは"),  # punctuation goes, ー stays
    ("人々♪〜・「」！？", "人々"),          # symbols go, 々 stays
    ("<|ja|><|NEUTRAL|>はい", "はい"),     # ASR tags removed
    ("  ", ""),
])
def test_normalize_ja(raw, expected):
    assert normalize_ja(raw) == expected


def test_edit_distance():
    assert edit_distance("きょうは", "きょうは") == 0
    assert edit_distance("きょうは", "きのうは") == 1
    assert edit_distance("abc", "") == 3
    assert edit_distance("", "abc") == 3


def test_corpus_cer_is_summed_over_clips():
    pairs = [("あいうえお", "あいうえお"), ("かきくけこ", "かきくけ"), ("さしす", "")]
    # 0 + 1 + 3 errors over 5 + 5 + 3 reference characters
    assert corpus_cer(pairs) == pytest.approx(4 / 13)
    assert corpus_cer([("、。", "")]) == 0.0  # nothing left to compare


# ---------------------------------------------------------------- local "Hub"

def _flac(seconds, sr=44100, channels=2):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, np.stack([tone] * channels, axis=1), sr, format="FLAC")
    return buf.getvalue()


AUDIO_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def _write_split(folder, split, files=2, rows_per_file=30, group=10, durations=None, texts=None):
    """Parquet files shaped like the Hub datasets (audio struct + transcription, small
    row groups). Row numbers run across the files in name order."""
    durations = durations or {}
    texts = texts or {}
    data = folder / "data"
    data.mkdir(parents=True, exist_ok=True)
    row = 0
    for f in range(files):
        audio, text = [], []
        for _ in range(rows_per_file):
            audio.append({"bytes": _flac(durations.get(row, 1.0)), "path": f"{row}.flac"})
            text.append(texts.get(row, f"これはテスト文の{row}番目です。"))
            row += 1
        table = pa.table({"audio": pa.array(audio, type=AUDIO_TYPE), "transcription": text})
        pq.write_table(table, data / f"{split}-{f:05d}-of-{files:05d}.parquet", row_group_size=group)


class LocalHub(ds.HubParquetSource):
    """HubParquetSource reading a local folder; counts group reads and can fail."""

    def __init__(self, root, split="test", revision="rev1", fail_after=None):
        self.root = root
        self.rev = revision
        self.opened = []
        self.group_reads = []
        self.fail_after = fail_after
        self.sleeps = []
        super().__init__("x/y", split, list_files=self._files, open_file=self._open,
                         get_revision=lambda: self.rev, sleep=self.sleeps.append)

    def _files(self, revision):
        return [f"data/{p.name}" for p in (self.root / "data").glob("*.parquet")] + ["README.md"]

    def _open(self, revision, path):
        self.opened.append(revision)
        if self.fail_after is not None and len(self.opened) > self.fail_after:
            raise OSError("connection reset")
        return open(self.root / path, "rb")

    def read_group(self, revision, file, group):
        self.group_reads.append((file, group))
        return super().read_group(revision, file, group)


@pytest.fixture
def bench_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVETRANSLATE_BENCH_DIR", str(tmp_path / "bench"))
    return tmp_path


def _expected_rows(hub, key, n):
    layout = hub.layout("rev1")
    base, total = [], 0
    for g in layout:
        base.append(total)
        total += g["rows"]
    rows = []
    for gi, offsets in ds.sample_plan(key, layout):
        rows += [base[gi] + o for o in offsets]
    return rows[:n + ds.WARMUP_CLIPS]


# ---------------------------------------------------------------- sampling

def test_layout_reads_every_row_group_of_the_split(tmp_path):
    _write_split(tmp_path, "test")
    _write_split(tmp_path, "other", files=1)  # another split in the same folder is ignored
    layout = LocalHub(tmp_path).layout("rev1")
    assert len(layout) == 6 and all(g["rows"] == 10 for g in layout)
    assert [g["file"] for g in layout][:3] == ["data/test-00000-of-00002.parquet"] * 3


def test_sample_plan_is_fixed_and_draws_per_group():
    layout = [{"file": "f", "group": i, "rows": 100, "bytes": 1} for i in range(50)]
    a = list(ds.sample_plan("anime", layout))
    assert a == list(ds.sample_plan("anime", layout))
    assert a != list(ds.sample_plan("vtuber", layout))
    assert all(len(rows) == ds.PER_GROUP and rows == sorted(rows) for _gi, rows in a)
    assert sorted(gi for gi, _ in a) == list(range(50))
    small = [{"file": "f", "group": 0, "rows": 3, "bytes": 1}]
    assert len(list(ds.sample_plan("x", small))[0][1]) == 3


def test_sampling_does_not_depend_on_hash_seed(monkeypatch):
    layout = [{"file": "f", "group": i, "rows": 100, "bytes": 1} for i in range(10)]
    first = list(ds.sample_plan("anime", layout))
    monkeypatch.setenv("PYTHONHASHSEED", "123")
    assert list(ds.sample_plan("anime", layout)) == first


def test_download_size_estimate():
    layout = [{"file": "f", "group": i, "rows": 100, "bytes": 8_000_000} for i in range(100)]
    # (200 + 3) clips * 1.15 / 8 per group -> 30 groups of 8 MB
    assert ds.estimate_mb(layout, 200) == pytest.approx(240.0)
    assert ds.estimate_mb([], 200) == 0.0


# ---------------------------------------------------------------- download

def test_download_writes_16k_mono_clips_reading_each_group_once(bench_dir):
    _write_split(bench_dir, "test")
    hub = LocalHub(bench_dir)
    infos = []
    m = ds.download_dataset("vtuber", hub, n=12, info=infos.append)
    assert m["complete"] and len(m["warmup"]) == ds.WARMUP_CLIPS and len(m["clips"]) == 12
    assert m["revision"] == "rev1" and m["method"] == ds.METHOD
    assert [c["row"] for c in m["warmup"] + m["clips"]] == _expected_rows(hub, "vtuber", 12)
    assert len(hub.group_reads) == len(set(hub.group_reads)) == 2  # 15 clips, 8 per group
    assert infos and "MB" in infos[0]
    first = m["clips"][0]
    folder = bench_dir / "bench" / "bench_clips" / "vtuber"
    audio, sr = sf.read(str(folder / f"{first['id']}.wav"))
    assert sr == 16000 and audio.ndim == 1 and abs(len(audio) / sr - 1.0) < 0.01
    assert (folder / f"{first['id']}.txt").read_text(encoding="utf-8") == first["ref"]
    warmup, clips = ds.load_clips("vtuber")
    assert len(warmup) == 3 and [c.id for c in clips] == [c["id"] for c in m["clips"]]


def test_unusable_rows_are_skipped_with_a_reason(bench_dir):
    _write_split(bench_dir, "test")
    hub = LocalHub(bench_dir)
    order = _expected_rows(hub, "vtuber", 2)
    _write_split(bench_dir, "test", durations={order[0]: 0.3, order[1]: 15.0}, texts={order[2]: "。"})
    m = ds.download_dataset("vtuber", LocalHub(bench_dir), n=2)
    reasons = {s["row"]: s["reason"] for s in m["skipped"]}
    assert "duration" in reasons[order[0]] and "duration" in reasons[order[1]]
    assert reasons[order[2]] == "reference too short"
    assert [c["row"] for c in m["warmup"] + m["clips"]] == order[3:5] + _expected_rows(hub, "vtuber", 5)[5:8]


def test_network_failure_stops_and_a_rerun_continues_at_the_same_row(bench_dir):
    _write_split(bench_dir, "test")
    expected = _expected_rows(LocalHub(bench_dir), "vtuber", 12)
    # the first opens (2 footers + 1 group) work, then the connection drops for good
    broken = LocalHub(bench_dir, fail_after=3)
    with pytest.raises(ds.DownloadError, match="connection reset"):
        ds.download_dataset("vtuber", broken, n=12)
    assert broken.sleeps  # it retried with back-off first
    saved = json.loads(ds.manifest_path("vtuber").read_text(encoding="utf-8"))
    done = [c["row"] for c in saved["warmup"] + saved["clips"]]
    assert done == expected[:8] and not saved["skipped"]
    again = LocalHub(bench_dir)
    m = ds.download_dataset("vtuber", again, n=12)
    assert [c["row"] for c in m["warmup"] + m["clips"]] == expected
    assert len(again.group_reads) == 1  # the finished group is not downloaded again


def test_revision_is_pinned_at_the_first_download(bench_dir):
    _write_split(bench_dir, "test")
    first = LocalHub(bench_dir, fail_after=3)
    with pytest.raises(ds.DownloadError):
        ds.download_dataset("vtuber", first, n=12)
    newer = LocalHub(bench_dir, revision="rev2")
    ds.download_dataset("vtuber", newer, n=12)
    assert set(newer.opened) == {"rev1"}  # the files of the first revision, not the new one


def test_unexpected_columns_stop_without_retrying(bench_dir):
    data = bench_dir / "data"
    data.mkdir()
    pq.write_table(pa.table({"text": ["a"] * 5}), data / "test-00000-of-00001.parquet", row_group_size=5)
    hub = LocalHub(bench_dir)
    with pytest.raises(ds.DownloadError, match="unexpected columns"):
        ds.download_dataset("vtuber", hub, n=1)
    assert hub.sleeps == []


def test_manifest_of_the_old_sampling_method_is_replaced(bench_dir):
    folder = bench_dir / "bench" / "bench_clips" / "anime"
    folder.mkdir(parents=True)
    old = {"hf_id": ds.DATASETS["anime"]["hf_id"], "split": "sfw", "seed": ds.SEED, "n_target": 200,
           "warmup": [{"id": "000001", "row": 1, "duration": 1.0, "ref": "x"}], "clips": [], "skipped": [],
           "complete": False}
    (folder / "manifest.json").write_text(json.dumps(old), encoding="utf-8")
    assert ds.dataset_status("anime") == (0, 200)
    assert ds.load_clips("anime") == ([], [])
    _write_split(bench_dir, "sfw")
    m = ds.download_dataset("anime", LocalHub(bench_dir, split="sfw"), n=2)
    assert m["method"] == ds.METHOD and len(m["warmup"]) == 3 and len(m["clips"]) == 2
    assert all(c["ref"] != "x" for c in m["warmup"] + m["clips"])


def test_complete_dataset_is_not_fetched_again(bench_dir):
    _write_split(bench_dir, "test")
    ds.download_dataset("baseline", LocalHub(bench_dir), n=2)
    again = LocalHub(bench_dir)
    ds.download_dataset("baseline", again, n=2)
    assert again.opened == []


def test_undecodable_audio_is_a_content_skip():
    with pytest.raises(ValueError, match="undecodable"):
        ds.decode_to_16k(b"not audio at all")


def test_live_recordings_are_found_by_folder(bench_dir):
    live = bench_dir / "bench" / "bench_clips" / "live"
    (live / "anime").mkdir(parents=True)
    (live / "stream.flac").write_bytes(_flac(1.0))
    (live / "anime" / "ep1.wav").write_bytes(b"x")
    (live / "notes.txt").write_text("ignored", encoding="utf-8")
    found = {(p.name, c) for p, c in ds.live_recordings()}
    assert found == {("stream.flac", "vtuber"), ("ep1.wav", "anime")}
