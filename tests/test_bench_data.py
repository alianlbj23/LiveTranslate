"""bench.textnorm and bench.datasets: Japanese CER and the seeded Hugging Face sample.
No network: httpx.MockTransport stands in for datasets-server."""
import io
import json

import httpx
import numpy as np
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


# ---------------------------------------------------------------- sampling

def test_sampling_is_fixed_per_dataset():
    a = ds.sample_candidates("anime", 271788, 203)
    assert a == ds.sample_candidates("anime", 271788, 203)
    assert a != ds.sample_candidates("vtuber", 271788, 203)
    assert len(a) == 609 and len(set(a)) == 609
    assert len(ds.sample_candidates("baseline", 50, 203)) == 50  # never more than the dataset


def test_sampling_does_not_depend_on_hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "123")
    assert ds.sample_candidates("anime", 1000, 10)[:5] == ds.sample_candidates("anime", 1000, 10)[:5]


# ---------------------------------------------------------------- download

def _flac(seconds, sr=44100, channels=2):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    audio = np.stack([tone] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="FLAC")
    return buf.getvalue()


class _FakeHub:
    """datasets-server stand-in. durations[row] sets each clip's length (default 2 s)."""

    def __init__(self, num_rows=40, durations=None, texts=None, fail_rows=(), rate_limit_once=False):
        self.num_rows = num_rows
        self.durations = durations or {}
        self.texts = texts or {}
        self.fail_rows = set(fail_rows)
        self.rate_limit_once = rate_limit_once
        self.row_calls = []

    def handler(self, request):
        url = request.url
        if url.path == "/size":
            return httpx.Response(200, json={"size": {"splits": [
                {"config": "default", "split": s, "num_rows": self.num_rows} for s in ("sfw", "test")]}})
        if url.host == "huggingface.co":
            return httpx.Response(200, json={"sha": "abc123"})
        if url.path == "/rows":
            row = int(url.params["offset"])
            self.row_calls.append(row)
            if self.rate_limit_once:
                self.rate_limit_once = False
                return httpx.Response(429, headers={"Retry-After": "3"})
            if row in self.fail_rows:
                return httpx.Response(503)
            return httpx.Response(200, json={"rows": [{"row": {
                "audio": [{"src": f"https://assets.example/{row}.flac", "type": "audio/flac"}],
                "transcription": self.texts.get(row, f"これはテスト文の{row}番目です。")}}]})
        if url.host == "assets.example":
            row = int(url.path.strip("/").split(".")[0])
            return httpx.Response(200, content=_flac(self.durations.get(row, 2.0)))
        return httpx.Response(404)


def _client(hub, sleeps=None):
    http = httpx.Client(transport=httpx.MockTransport(hub.handler))
    return ds.RowsClient(http, sleep=(sleeps.append if sleeps is not None else (lambda s: None)),
                         min_interval=0)


@pytest.fixture
def bench_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVETRANSLATE_BENCH_DIR", str(tmp_path))
    return tmp_path


def test_download_writes_16k_mono_clips_and_a_manifest(bench_dir):
    hub = _FakeHub()
    m = ds.download_dataset("vtuber", _client(hub), n=4)
    assert m["complete"] and len(m["warmup"]) == ds.WARMUP_CLIPS and len(m["clips"]) == 4
    expected_rows = ds.sample_candidates("vtuber", 40, 4 + ds.WARMUP_CLIPS)[:7]
    assert [c["row"] for c in m["warmup"] + m["clips"]] == expected_rows
    first = m["clips"][0]
    audio, sr = sf.read(str(bench_dir / "bench_clips" / "vtuber" / f"{first['id']}.wav"))
    assert sr == 16000 and audio.ndim == 1 and abs(len(audio) / sr - 2.0) < 0.01
    assert (bench_dir / "bench_clips" / "vtuber" / f"{first['id']}.txt").read_text(encoding="utf-8") == first["ref"]
    warmup, clips = ds.load_clips("vtuber")
    assert len(warmup) == 3 and [c.id for c in clips] == [c["id"] for c in m["clips"]]


def test_clips_outside_the_length_limits_are_skipped_with_a_reason(bench_dir):
    order = ds.sample_candidates("vtuber", 40, 5)
    hub = _FakeHub(durations={order[0]: 0.3, order[1]: 15.0}, texts={order[2]: "。"})
    m = ds.download_dataset("vtuber", _client(hub), n=2)
    reasons = {s["row"]: s["reason"] for s in m["skipped"]}
    assert "duration" in reasons[order[0]] and "duration" in reasons[order[1]]
    assert reasons[order[2]] == "reference too short"
    assert [c["row"] for c in m["warmup"] + m["clips"]] == order[3:8]


def test_network_failure_stops_and_a_rerun_continues_at_the_same_row(bench_dir):
    order = ds.sample_candidates("vtuber", 40, 5)
    hub = _FakeHub(fail_rows={order[3]})
    client = ds.RowsClient(httpx.Client(transport=httpx.MockTransport(hub.handler)),
                           sleep=lambda s: None, min_interval=0, retries=1)
    with pytest.raises(ds.DownloadError):
        ds.download_dataset("vtuber", client, n=2)
    saved = json.loads(ds.manifest_path("vtuber").read_text(encoding="utf-8"))
    assert [c["row"] for c in saved["warmup"] + saved["clips"]] == order[:3]
    assert not saved["skipped"]  # a network failure is not a content skip
    hub.fail_rows.clear()
    hub.row_calls.clear()
    m = ds.download_dataset("vtuber", client, n=2)
    assert [c["row"] for c in m["warmup"] + m["clips"]] == order[:5]
    assert hub.row_calls == order[3:5]  # nothing fetched twice


def test_rate_limit_waits_for_retry_after(bench_dir):
    sleeps = []
    hub = _FakeHub(rate_limit_once=True)
    ds.download_dataset("vtuber", _client(hub, sleeps), n=1)
    assert 3.0 in sleeps


def test_client_error_is_not_retried(bench_dir):
    def handler(request):
        return httpx.Response(404, text="nope")
    client = ds.RowsClient(httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None, min_interval=0)
    with pytest.raises(ds.DownloadError, match="HTTP 404"):
        client.num_rows("x/y", "test")


def test_complete_dataset_is_not_fetched_again(bench_dir):
    hub = _FakeHub()
    ds.download_dataset("baseline", _client(hub), n=2)
    hub.row_calls.clear()
    ds.download_dataset("baseline", _client(hub), n=2)
    assert hub.row_calls == []


def test_undecodable_audio_is_a_content_skip():
    with pytest.raises(ValueError, match="undecodable"):
        ds.decode_to_16k(b"not audio at all")


def test_live_recordings_are_found_by_folder(bench_dir):
    live = bench_dir / "bench_clips" / "live"
    (live / "anime").mkdir(parents=True)
    (live / "stream.flac").write_bytes(_flac(1.0))
    (live / "anime" / "ep1.wav").write_bytes(b"x")
    (live / "notes.txt").write_text("ignored", encoding="utf-8")
    found = {(p.name, c) for p, c in ds.live_recordings()}
    assert found == {("stream.flac", "vtuber"), ("ep1.wav", "anime")}
