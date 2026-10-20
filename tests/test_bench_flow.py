"""End-to-end benchmark runs with fake ASR workers, translators, judges and Ollama: stage
order, rankings, skipped models, cross judging, report output and resume."""
import io
import json

import numpy as np
import pytest
import soundfile as sf

from bench import analysis, datasets, llm_servers
from bench.orchestrator import Bench, Deps
from bench.state import RunStore

SR = 16000
STEP = 0.002
REFS = ["今日はいい天気ですね", "本当にありがとうございました", "早く行こうよ", "それは違うと思う",
        "明日また会おうね", "お腹すいたなあ", "ちょっと待ってください", "どうしてそうなるの"]


def _entry(name, model, **kw):
    e = {"name": name, "model": model, "api_base": "http://127.0.0.1:11434/v1", "api_key": "ollama",
         "proxy": "none", "thinking_style": "openai"}
    e.update(kw)
    return e


MODELS = [_entry("A Model", "model-a"), _entry("B Model", "model-b"),
          _entry("Sakura", "sakura", prompt_style="sakura", thinking_style="off")]
FP = {m["name"]: llm_servers.fingerprint(m) for m in MODELS}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Datasets on disk; each clip's audio level encodes a unique id -> (dataset, ref)."""
    monkeypatch.setenv("LIVETRANSLATE_BENCH_DIR", str(tmp_path))
    by_id = {}
    k = 0
    counts = {"anime": 4, "vtuber": 4, "baseline": 2}
    for key, n in counts.items():
        folder = tmp_path / "bench_clips" / key
        folder.mkdir(parents=True)
        entries = []
        for i in range(n + datasets.WARMUP_CLIPS):
            k += 1
            cid = f"{i:06d}"
            sf.write(str(folder / f"{cid}.wav"), np.full(SR, k * STEP, dtype=np.float32), SR, subtype="PCM_16")
            ref = REFS[(k - 1) % len(REFS)]
            by_id[k] = (key, ref)
            entries.append({"id": cid, "row": i, "duration": 1.0, "ref": ref})
        spec = datasets.DATASETS[key]
        (folder / "manifest.json").write_text(json.dumps({
            "hf_id": spec["hf_id"], "split": spec["split"], "seed": datasets.SEED, "n_target": n,
            "warmup": entries[:datasets.WARMUP_CLIPS], "clips": entries[datasets.WARMUP_CLIPS:],
            "skipped": [], "complete": True}, ensure_ascii=False), encoding="utf-8")
    return {"by_id": by_id, "counts": counts, "dir": tmp_path}


def _decode(by_id, audio):
    return by_id[int(round(float(np.mean(audio)) / STEP))]


class FakeASR:
    def __init__(self, config, by_id):
        self.config = config
        self.by_id = by_id

    def start(self):
        if self.config["engine_type"] == "whisper":
            raise RuntimeError("CUDA out of memory")

    def wait_ready(self):
        return {}

    def transcribe(self, audio):
        dataset, ref = _decode(self.by_id, audio)
        kind = self.config["engine_type"], self.config.get("funasr_model")
        if kind == ("funasr", "sensevoice-small"):
            text = ref
        elif kind[0] == "funasr":           # MLT-Nano: drops the last two characters
            text = ref[:-2]
        else:                               # Anime-Whisper: one extra char on anime, nothing on TV speech
            text = ref + "ね" if dataset == "anime" else ""
        return {"text": text, "language": "ja"}

    def shutdown(self):
        pass


class FakeTranslator:
    def __init__(self, entry, explode_after=None):
        self.prefix = entry["name"][0]
        self.explode_after = explode_after
        self.calls = 0

    def translate_iter(self, text, source_language="ja"):
        self.calls += 1
        if self.explode_after is not None and self.calls > self.explode_after:
            raise KeyboardInterrupt
        yield f"{self.prefix}譯"
        yield f"{self.prefix}譯：{len(text)}字"


class FakeJudge:
    """Model A translations deserve 8, B 6, Sakura 7, anything else 5. Judge A is one
    point generous, judge B one point strict; cross judging must cancel that out."""

    def __init__(self, entry):
        self.bias = {"A Model": 1, "B Model": -1}[entry["name"]]

    def score(self, category, ja, tl):
        base = {"A": 8, "B": 6, "S": 7}.get(tl[:1], 5)
        v = max(1, min(10, base + self.bias))
        return {"accuracy": v, "naturalness": v, "tone": v, "taiwan": v, "note": ""}

    def vad(self, continuous, lines):
        return {"hallucinated": [], "missed": []}


class FakeOllama:
    def __init__(self, root):
        self.root = root

    def alive(self):
        return True

    def installed(self):
        return {m["model"] for m in MODELS}

    def loaded(self):
        return []

    def unload_all(self, timeout=30.0, sleep=None):
        return True

    def load(self, model, timeout=300.0):
        return 0.01


class FakeVram:
    def used_mb(self):
        return 2000.0


def _plan(world, **kw):
    plan = {"schema": 1, "stages": [1, 3], "datasets": ["anime", "vtuber", "baseline"],
            "clip_counts": dict(world["counts"]), "asr": ["sensevoice", "mlt-nano", "anime-whisper", "whisper-turbo"],
            "download_missing": False, "reuse_stage1_from": None, "recordings": [],
            "tl_models": [{"fp": FP[m["name"]], "name": m["name"], "model": m["model"], "kind": "ollama"} for m in MODELS],
            "judges": [FP["A Model"], FP["B Model"]], "judge_mode": "cross", "together_clips": 2,
            "restart_ollama": False}
    plan.update(kw)
    return plan


SETTINGS = {"models": MODELS, "asr_language": "ja", "hub": "hf", "target_language": "zh"}
CONFIG = {"asr": {"compute_type": "float16"}, "translation": {"max_tokens": 256, "temperature": 0.3}}


def _deps(world, translator=None, app_running=None, is_cached=None):
    return Deps(asr_client_factory=lambda cfg: FakeASR(cfg, world["by_id"]),
                translator_factory=translator or (lambda e, s, c: FakeTranslator(e)),
                judge_factory=FakeJudge, ollama_factory=FakeOllama, vram=FakeVram(),
                app_running=app_running or (lambda: []), sleep=lambda s: None, out=io.StringIO(),
                is_cached=is_cached or (lambda cand: True), together_realtime=False)


def test_full_run_ranks_skips_cross_judges_and_reports(world):
    store = RunStore.create(_plan(world))
    Bench(store, SETTINGS, CONFIG, _deps(world)).run()
    assert store.state["status"] == "done"

    rankings = store.frozen("rankings")
    assert rankings["anime"][0] == "sensevoice" and rankings["vtuber"][0] == "sensevoice"
    assert "whisper-turbo" not in rankings["anime"]
    failed = [m for m in store.read("asr_meta") if m["asr"] == "whisper-turbo"]
    assert failed and failed[-1]["status"] == "failed" and "CUDA out of memory" in failed[-1]["reason"]
    assert store.frozen("top2")["anime"] == ["sensevoice", "anime-whisper"]
    assert store.frozen("top2")["vtuber"] == ["sensevoice", "mlt-nano"]

    judged = store.read("judge")
    assert judged and not [r for r in judged if r["candidate"] and r["candidate"] == r["judge"]]
    quality = analysis.quality_table(store)
    assert quality[(FP["A Model"], "anime")]["quality"] == pytest.approx(8.0)
    assert quality[(FP["B Model"], "anime")]["quality"] == pytest.approx(6.0)
    assert quality[(FP["Sakura"], "anime")]["quality"] == pytest.approx(7.0)

    together = store.read("together")
    pairs = {(r["asr"], r["model"]) for r in together}
    assert ("anime-whisper", FP["A Model"]) in pairs and ("mlt-nano", FP["B Model"]) in pairs
    assert not [r for r in together if r["asr"] == "whisper-turbo"]

    html = (store.dir / "report.html").read_text(encoding="utf-8")
    assert "推薦組合" in html and "Anime-Whisper 的訓練資料" in html and "A Model" in html
    recs = json.loads((store.dir / "recommendations.json").read_text(encoding="utf-8"))
    assert {r["id"][0] for r in recs} == {"A", "V"}
    best_quality = next(r for r in recs if r["category"] == "anime" and "品質最好" in r["kinds"])
    assert best_quality["tl_name"] == "A Model"
    csv_text = (store.dir / "results.csv").read_text(encoding="utf-8-sig")
    assert csv_text.splitlines()[0].startswith("category,asr,vad_threshold")
    assert len(csv_text.splitlines()) == 1 + 2 * 2 * 3  # 2 categories x 2 ASRs x 3 models


def test_interrupted_run_resumes_without_duplicates(world):
    store = RunStore.create(_plan(world))
    exploding = {"n": 0}

    def factory(entry, s, c):
        exploding["n"] += 1
        # the first translator dies after its warm-up and 2 sentences
        return FakeTranslator(entry, explode_after=5 if exploding["n"] == 1 else None)

    with pytest.raises(KeyboardInterrupt):
        Bench(store, SETTINGS, CONFIG, _deps(world, translator=factory)).run()
    assert RunStore(store.dir).state["status"] == "interrupted"
    partial = len(RunStore(store.dir).read("tl"))
    assert 0 < partial < 3 * 8

    resumed = RunStore(store.dir)
    Bench(resumed, SETTINGS, CONFIG, _deps(world)).run()
    tl = resumed.read("tl")
    keys = [(r["model"], r["category"], r["clip"]) for r in tl]
    assert len(keys) == len(set(keys)) == 3 * 8
    asr_keys = [(r["asr"], r["dataset"], r["clip"]) for r in resumed.read("asr")]
    assert len(asr_keys) == len(set(asr_keys))
    assert resumed.state["status"] == "done"


def test_model_not_downloaded_is_skipped_with_a_reason(world):
    store = RunStore.create(_plan(world, asr=["sensevoice", "mlt-nano"], stages=[1]))
    Bench(store, SETTINGS, CONFIG, _deps(world, is_cached=lambda c: c.id != "mlt-nano")).run()
    meta = {m["asr"]: m for m in store.read("asr_meta")}
    assert meta["mlt-nano"]["status"] == "failed" and "尚未下載" in meta["mlt-nano"]["reason"]
    assert store.frozen("rankings")["anime"] == ["sensevoice"]


def test_run_pauses_while_livetranslate_is_open(world):
    calls = {"n": 0}

    def running():
        calls["n"] += 1
        return [4242] if calls["n"] <= 2 else []

    store = RunStore.create(_plan(world, asr=["sensevoice"], stages=[1]))
    deps = _deps(world, app_running=running)
    slept = []
    deps.sleep = slept.append
    Bench(store, SETTINGS, CONFIG, deps).run()
    assert slept.count(5.0) >= 2
    assert "評測暫停" in deps.out.getvalue()


def test_translation_model_failing_every_sentence_is_marked_failed(world):
    class Broken:
        def translate_iter(self, text, source_language="ja"):
            raise ConnectionError("refused")
            yield  # pragma: no cover

    store = RunStore.create(_plan(world, datasets=["anime"], clip_counts={"anime": 4}))
    Bench(store, SETTINGS, CONFIG, _deps(world, translator=lambda e, s, c: Broken())).run()
    tl = store.read("tl")
    assert tl and all(not r["ok"] and r["error"] == "connection" for r in tl if r["source"])
    combos = analysis.combos(store)["anime"]
    assert all(c["quality"] == pytest.approx(1.0) for c in combos)
