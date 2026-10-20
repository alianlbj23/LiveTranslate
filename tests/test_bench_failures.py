"""Failures the user must be able to see (review of the first benchmark version): missing
judge coverage, failed VAD judging, leaked ASR workers, too few clips, fatal errors,
retried items, stage 1 results overwritten by later stages, failures under load, and
unfinished runs. Reuses the fakes of test_bench_flow."""
import json

import numpy as np
import pytest

from bench import analysis, cli, settings_io, state
from bench.orchestrator import Bench
from bench.state import RunStore
from test_bench_flow import CONFIG, FP, MODELS, SETTINGS, FakeASR, FakeJudge, FakeTranslator, _deps, _plan, make_world


@pytest.fixture
def world(tmp_path, monkeypatch):
    return make_world(tmp_path, monkeypatch)


def _run(store, deps):
    Bench(store, SETTINGS, CONFIG, deps).run()
    return RunStore(store.dir)


# ---------------------------------------------------------------- C1 judge coverage

def test_quality_without_enough_judge_scores_is_not_reported(world):
    class OnlyAJudge(FakeJudge):
        def __init__(self, entry):
            super().__init__(entry)
            if entry["name"] == "A Model":
                raise RuntimeError("judge cannot start")

    deps = _deps(world)
    deps.judge_factory = OnlyAJudge
    store = _run(RunStore.create(_plan(world)), deps)
    quality = analysis.quality_table(store)
    # B is judged only by judge A, which failed: no quality instead of "1.0 from failures only"
    q = quality[(FP["B Model"], "anime")]
    assert q["quality"] is None and q["unjudged"] == 4 and "評審覆蓋不足" in q["note"]
    combo = next(c for c in analysis.combos(store)["anime"] if c["tl"] == FP["B Model"])
    assert not combo["recommended"] and any("評審覆蓋不足" in r for r in combo["reasons"])
    judge_task = store.task(f"s3judge/{FP['A Model']}")
    assert judge_task["status"] == "failed" and "judge cannot start" in judge_task["reason"]


# ---------------------------------------------------------------- C2 / C3 VAD judging

@pytest.fixture
def recording(world):
    live = world["dir"] / "bench_clips" / "live"
    live.mkdir(parents=True)
    import soundfile as sf

    sr = 16000
    audio = np.zeros(sr * 4, dtype=np.float32)
    audio[sr:sr * 2] = 0.3
    path = live / "stream.wav"
    sf.write(str(path), audio, sr)
    return {"path": str(path), "category": "vtuber", "minutes": 4 / 60}


class _Vad:
    def __init__(self, th, settings):
        self._speech_samples = 0

    def _get_effective_silence_limit(self):
        return 1

    def process_chunk(self, chunk):
        return None

    def flush(self):
        return np.full(16000, 0.002, dtype=np.float32)  # decoded as clip id 1 by FakeASR


def _load_audio(path):
    """The recording decodes as clip id 1 for FakeASR; dataset clips load normally."""
    from bench import datasets

    if str(path).endswith("stream.wav"):
        return np.full(16000, 0.002, dtype=np.float32)
    return datasets.load_audio_file(path)


def test_vad_judging_failures_are_not_shown_as_zero_errors(world, recording):
    class NoVadJudge(FakeJudge):
        def vad(self, continuous, lines):
            return None

    deps = _deps(world)
    deps.judge_factory = NoVadJudge
    deps.vad_factory = _Vad
    deps.load_audio = _load_audio
    store = _run(RunStore.create(_plan(world, stages=[1, 2, 3], recordings=[recording])), deps)
    assert analysis.best_vad_thresholds(store) == {}
    table = analysis.vad_table(store)
    assert table and all(v["judged_windows"] == 0 and v["errors_per_min"] is None
                         for by_th in table.values() for v in by_th.values())
    assert all(not c["vad_measured"] for rows in analysis.combos(store).values() for c in rows)
    html = (store.dir / "report.html").read_text(encoding="utf-8")
    assert "評審失敗" in html and "沒有自己的錄音" not in html


def test_vad_is_judged_by_the_first_judge_that_loads(world, recording):
    class FirstJudgeBroken(FakeJudge):
        def __init__(self, entry):
            super().__init__(entry)
            if entry["name"] == "A Model":
                raise RuntimeError("cannot start")

    deps = _deps(world)
    deps.judge_factory = FirstJudgeBroken
    deps.vad_factory = _Vad
    deps.load_audio = _load_audio
    store = _run(RunStore.create(_plan(world, stages=[1, 2, 3], recordings=[recording])), deps)
    assert store.read("vad") and all(r["judged_windows"] > 0 for r in store.read("vad"))


# ---------------------------------------------------------------- C4 worker leak

def test_worker_error_does_not_leave_a_second_model_loaded(world):
    from asr_client import ASRWorkerError

    live = {"n": 0, "max": 0}

    class Leaky(FakeASR):
        calls = 0

        def start(self):
            live["n"] += 1
            live["max"] = max(live["max"], live["n"])

        def transcribe(self, audio):
            Leaky.calls += 1
            if Leaky.calls == 5:
                raise ASRWorkerError({"message": "CUDA error", "recoverable": True})
            return super().transcribe(audio)

        def shutdown(self):
            live["n"] -= 1

    deps = _deps(world)
    deps.asr_client_factory = lambda cfg: Leaky(cfg, world["by_id"])
    store = _run(RunStore.create(_plan(world, asr=["sensevoice"], stages=[1])), deps)
    assert live["max"] == 1 and live["n"] == 0
    errors = [r for r in store.read("asr") if r.get("error")]
    assert len(errors) == 1 and "CUDA error" in errors[0]["error"]


# ---------------------------------------------------------------- C5 too few clips

class _NoOllama:
    def __init__(self, root):
        pass

    def alive(self):
        return False


def test_plan_uses_the_clips_that_exist(world, monkeypatch):
    monkeypatch.setattr(cli, "_model_options", lambda s: [(m, True, "已下載") for m in MODELS])
    monkeypatch.setattr(cli, "is_cached", lambda c, hub: True)
    monkeypatch.setattr(cli.llm_servers, "Ollama", _NoOllama)
    answers = iter(["", "", "", "", ""])
    out = []
    plan = cli.build_plan(cli.Console(input_fn=lambda p: next(answers), print_fn=out.append), SETTINGS)
    assert plan["clip_counts"] == {"anime": 4, "vtuber": 4, "baseline": 2}
    assert any("只有 4 句" in line for line in out)


def test_category_without_a_ranked_asr_is_reported_not_silently_skipped(world):
    class AllFail(FakeASR):
        def start(self):
            raise RuntimeError("no GPU")

    deps = _deps(world)
    deps.asr_client_factory = lambda cfg: AllFail(cfg, world["by_id"])
    store = _run(RunStore.create(_plan(world)), deps)
    tl_tasks = [t for t in store.state["tasks"] if t["id"].startswith("s3tl/")]
    assert tl_tasks and all(t["status"] == "failed" and "沒有可用的辨識結果" in t["reason"] for t in tl_tasks)
    assert "沒有任何辨識模型完成" in deps.out.getvalue()


def test_reusing_stage1_requires_a_complete_matching_run(world, monkeypatch):
    old = RunStore.create(_plan(world, asr=["sensevoice"]))
    old.set_task("s1/sensevoice", status="done")
    monkeypatch.setattr(cli, "_model_options", lambda s: [(m, True, "已下載") for m in MODELS])
    answers = iter(["1", "", ""])  # untick stage 1 and confirm, datasets: confirm
    out = []
    plan = cli.build_plan(cli.Console(input_fn=lambda p: next(answers), print_fn=out.append), SETTINGS)
    assert plan is None and any("沒有完整的第 1 階段結果" in line for line in out)


# ---------------------------------------------------------------- H1 fatal errors

def test_translator_that_cannot_be_built_fails_only_its_task(world):
    def factory(entry, s, c):
        if entry["name"] == "B Model":
            raise KeyError("api_key")
        return FakeTranslator(entry)

    store = _run(RunStore.create(_plan(world)), _deps(world, translator=factory))
    assert store.state["status"] == "done"
    task = store.task(f"s3tl/{FP['B Model']}")
    assert task["status"] == "failed" and "api_key" in task["reason"]
    assert store.task(f"s3tl/{FP['A Model']}")["status"] == "done"


def test_unexpected_error_marks_the_run_and_cli_points_to_the_log(world, monkeypatch):
    store = RunStore.create(_plan(world, asr=["sensevoice"], stages=[1]))

    def boom(self):
        raise RuntimeError("disk full")

    monkeypatch.setattr(Bench, "stage1", boom)
    out = []
    monkeypatch.setattr(settings_io, "find_running_app", lambda: [])
    code = cli._run(cli.Console(input_fn=lambda p: "", print_fn=out.append), store, SETTINGS, CONFIG)
    assert code == 1
    again = RunStore(store.dir)
    assert again.state["status"] == "error" and "disk full" in again.state["error"]
    assert any("bench.log" in line for line in out)
    assert "disk full" in (store.dir / "bench.log").read_text(encoding="utf-8")


def test_ollama_errors_while_unloading_do_not_stop_the_run(world):
    import httpx

    from test_bench_flow import FakeOllama

    class Flaky(FakeOllama):
        def unload_all(self, timeout=30.0, sleep=None):
            raise httpx.ReadTimeout("busy")

    deps = _deps(world)
    deps.ollama_factory = Flaky
    store = _run(RunStore.create(_plan(world, asr=["sensevoice"])), deps)
    assert store.state["status"] == "done"


# ---------------------------------------------------------------- H4 retries

def test_failed_sentences_are_retried_on_resume(world):
    flaky = {"fail": True}

    class Sometimes(FakeTranslator):
        def translate_iter(self, text, source_language="ja"):
            if flaky["fail"] and self.calls >= 3:  # warm-up passes, real sentences fail
                self.calls += 1
                raise TimeoutError("slow")
            yield from super().translate_iter(text, source_language)

    store = _run(RunStore.create(_plan(world, tl_models=_plan(world)["tl_models"][:1], judges=[FP["A Model"]])),
                 _deps(world, translator=lambda e, s, c: Sometimes(e)))
    assert all(not r["ok"] for r in store.read("tl"))
    flaky["fail"] = False
    for t in store.state["tasks"]:
        if t["id"].startswith(("s3tl/", "s3judge/", "s3tog/")):
            store.set_task(t["id"], status="pending")
    store.state["status"] = "running"
    store.save()
    store = _run(store, _deps(world, translator=lambda e, s, c: Sometimes(e)))
    latest = {(r["category"], r["clip"]): r for r in store.read("tl")}
    assert latest and all(r["ok"] for r in latest.values())
    assert analysis.quality_table(store)[(FP["A Model"], "anime")]["quality"] > 5


# ---------------------------------------------------------------- H5 stage 1 meta

def test_later_stage_asr_failure_does_not_override_stage1(world):
    count = {"starts": 0}

    class FailsLater(FakeASR):
        def start(self):
            if self.config["engine_type"] == "whisper":
                raise RuntimeError("CUDA out of memory")
            if self.config.get("funasr_model") == "sensevoice-small":
                count["starts"] += 1
                if count["starts"] > 1:
                    raise RuntimeError("cannot load now")

    deps = _deps(world)
    deps.asr_client_factory = lambda cfg: FailsLater(cfg, world["by_id"])
    store = _run(RunStore.create(_plan(world)), deps)
    table = analysis.asr_table(store)
    assert table["anime"]["sensevoice"]["complete"] is True
    assert analysis.asr_rankings(store)["anime"][0] == "sensevoice"


# ---------------------------------------------------------------- H6 failures under load

def test_failures_while_loaded_together_count_in_latency(world):
    class SlowTogether(FakeTranslator):
        def translate_iter(self, text, source_language="ja"):
            import threading

            if threading.current_thread().name.startswith("bench-tl"):
                raise TimeoutError("busy GPU")
            yield from super().translate_iter(text, source_language)

    store = _run(RunStore.create(_plan(world)), _deps(world, translator=lambda e, s, c: SlowTogether(e)))
    rows = analysis.combos(store)["anime"]
    assert rows and all(not c["recommended"] for c in rows)
    assert all(c["together_fail_rate"] == 1.0 for c in rows)


# ---------------------------------------------------------------- H7 unfinished runs

def test_report_lists_failed_items_and_marks_unfinished_runs(world):
    store = _run(RunStore.create(_plan(world)), _deps(world))
    html = (store.dir / "report.html").read_text(encoding="utf-8")
    assert "略過或失敗的項目" in html and "CUDA out of memory" in html
    store.state["status"] = "interrupted"
    store.save()
    from bench import report

    report.build(store)
    html = (store.dir / "report.html").read_text(encoding="utf-8")
    assert "評測沒有跑完" in html
    recs = json.loads((store.dir / "recommendations.json").read_text(encoding="utf-8"))
    assert recs["status"] == "interrupted"


def test_apply_from_an_unfinished_run_needs_confirmation(world, monkeypatch, tmp_path):
    store = _run(RunStore.create(_plan(world)), _deps(world))
    store.state["status"] = "interrupted"
    store.save()
    from bench import report

    report.build(store)
    settings_path = tmp_path / "user_settings.json"
    settings_path.write_text(json.dumps({**SETTINGS, "asr_engine": "whisper", "vad_threshold": 0.5,
                                         "active_model": 0}), encoding="utf-8")
    monkeypatch.setattr(cli.settings_io.paths, "SETTINGS_FILE", settings_path)
    monkeypatch.setattr(settings_io, "find_running_app", lambda: [])
    answers = iter(["", "n"])  # pick the run, refuse the warning
    out = []
    assert cli.cmd_apply(cli.Console(input_fn=lambda p: next(answers), print_fn=out.append)) == 0
    assert any("沒有跑完" in line for line in out)
    assert json.loads(settings_path.read_text(encoding="utf-8"))["asr_engine"] == "whisper"


# ---------------------------------------------------------------- H8 apply that would be undone

def test_apply_warns_that_the_launcher_keeps_its_local_model(tmp_path, monkeypatch):
    settings = {"models": [{"name": "Cloud", "model": "gpt-x", "api_base": "https://api.example.com/v1", "api_key": "k"},
                           MODELS[0]], "asr_engine": "whisper", "vad_threshold": 0.5, "active_model": 1}
    path = tmp_path / "user_settings.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr(settings_io, "find_running_app", lambda: [])
    changes = settings_io.planned_changes(settings, "sensevoice", 0.5, settings["models"][0])
    notes = settings_io.apply_notes(settings["models"][0])
    assert notes and "啟動器" in notes[0]
    settings_io.apply_changes(changes, settings["models"][0], settings_path=path,
                              selected_file=tmp_path / "selected_model.txt")


def test_apply_rolls_back_when_the_launcher_selection_cannot_be_written(tmp_path, monkeypatch):
    import local_models

    settings = {"models": MODELS, "asr_engine": "whisper", "vad_threshold": 0.5, "active_model": 0}
    path = tmp_path / "user_settings.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr(settings_io, "find_running_app", lambda: [])
    monkeypatch.setattr(local_models, "remember_selected", lambda entry: False)
    changes = settings_io.planned_changes(settings, "sensevoice", 0.4, MODELS[1])
    with pytest.raises(settings_io.SettingsError, match="selected_model"):
        settings_io.apply_changes(changes, MODELS[1], settings_path=path, selected_file=tmp_path / "sel.txt")
    assert json.loads(path.read_text(encoding="utf-8")) == settings


# ---------------------------------------------------------------- H9 failures mid-run

def test_model_that_breaks_mid_run_is_stopped_and_reported(world):
    class DiesLater(FakeTranslator):
        def translate_iter(self, text, source_language="ja"):
            if self.calls >= 5:  # warm-up (3) + 2 good sentences, then the server is gone
                self.calls += 1
                raise ConnectionError("refused")
            yield from super().translate_iter(text, source_language)

    counts = {"anime": 4, "vtuber": 4, "baseline": 2}
    store = _run(RunStore.create(_plan(world, clip_counts=counts, tl_models=_plan(world)["tl_models"][:1],
                                       judges=[FP["A Model"]])),
                 _deps(world, translator=lambda e, s, c: DiesLater(e)))
    assert store.read("tl_meta")[-1]["errors"] == {"connection": 6}


def test_resume_takes_the_lock_before_touching_state(world, monkeypatch):
    store = RunStore.create(_plan(world))
    store.set_task("s1/sensevoice", status="failed", reason="x")
    monkeypatch.setattr(state, "running_pid", lambda: 4242)
    monkeypatch.setattr(settings_io, "find_running_app", lambda: [])
    out = []
    assert cli.cmd_resume(cli.Console(input_fn=lambda p: "y", print_fn=out.append)) == 1
    assert RunStore(store.dir).task("s1/sensevoice")["status"] == "failed"
    assert any("另一個評測" in line for line in out)



# ---------------------------------------------------------------- real-time replay timing

def test_realtime_replay_measures_each_sentence_when_it_finishes(world):
    """Review finding: end-to-end time was taken after every clip had been submitted, so
    early sentences looked as slow as the whole replay. Each translation must be timed on
    its own worker when it finishes."""
    import time

    from bench.datasets import Clip
    from bench.orchestrator import Deps

    class Session:
        def transcribe(self, audio):
            time.sleep(0.01)
            return {"text": "本当にそう思う", "language": "ja"}, 10.0

    class Slow:
        def translate_iter(self, text, source_language="ja"):
            time.sleep(0.1)
            yield "我真的這麼覺得"

    from bench import asr_runner

    asr_runner.warm_up_filters("ja")  # done by _run_together before the replay
    clips = [Clip("anime", f"{i:06d}", world["dir"] / "x.wav", "本当にそう思う", 0.3) for i in range(3)]
    store = RunStore.create(_plan(world))
    bench = Bench(store, SETTINGS, CONFIG, Deps(load_audio=lambda p: np.zeros(1600, dtype=np.float32),
                                                vram=_deps(world).vram, out=_deps(world).out))
    records = bench._replay(Session(), Slow(), clips, hint_missing=False)
    assert [r["ok"] for r in records] == [True, True, True]
    for r in records:
        assert 0.09 <= r["e2e_ms"] / 1000 < 0.4, r["e2e_ms"]
