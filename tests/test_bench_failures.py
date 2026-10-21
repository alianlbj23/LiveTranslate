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


# ---------------------------------------------------------------- judge coverage

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


# ---------------------------------------------------------------- VAD judging

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


def _vad_deps(world, judge_factory):
    deps = _deps(world)
    deps.judge_factory = judge_factory
    deps.vad_factory = _Vad
    deps.load_audio = _load_audio
    return deps


def _as_cmd_resume(store):
    """What cli.cmd_resume does before running again: judge tasks back to pending."""
    for t in store.state["tasks"]:
        if t["id"].startswith("s3judge/") or t["id"] == "s2judge":
            store.set_task(t["id"], status="pending", reason=None)
    store.state["status"] = "running"
    store.save()


def test_unjudged_vad_windows_are_judged_again_on_resume(world, recording):
    """Review finding: VAD records whose windows all failed counted as done, so a resume
    never judged them again and the run had no VAD result for good."""
    class NoVadJudge(FakeJudge):
        def vad(self, continuous, lines):
            return None

    store = _run(RunStore.create(_plan(world, stages=[1, 2, 3], recordings=[recording])), _vad_deps(world, NoVadJudge))
    assert analysis.best_vad_thresholds(store) == {}
    _as_cmd_resume(store)
    store = _run(store, _vad_deps(world, FakeJudge))
    latest = analysis._latest(store.read("vad"), ("asr", "threshold", "recording"))
    assert latest and all(r["judged_windows"] == r["windows"] > 0 for r in latest)
    assert analysis.best_vad_thresholds(store)


def test_vad_judge_that_never_answers_is_stopped_and_the_next_one_takes_over(world, recording, monkeypatch):
    from bench import orchestrator

    monkeypatch.setattr(orchestrator, "FAIL_STREAK", 2)
    calls = {"A Model": 0, "B Model": 0}

    class NoVadJudge(FakeJudge):
        def __init__(self, entry):
            super().__init__(entry)
            self.name = entry["name"]
            self.last_error = "AuthenticationError: 401 invalid api key"

        def vad(self, continuous, lines):
            calls[self.name] += 1
            return None

    store = _run(RunStore.create(_plan(world, stages=[1, 2, 3], recordings=[recording])), _vad_deps(world, NoVadJudge))
    assert calls == {"A Model": 2, "B Model": 2}
    task = store.task("s2judge")
    assert task["status"] == "failed" and "連續 2" in task["reason"] and "401 invalid api key" in task["reason"]


def test_stage2_starts_a_fresh_worker_after_one_died_on_a_recording(world, recording):
    """Review finding: after the worker died on one recording the session had no client,
    and every later recording failed with "'NoneType' object has no attribute"."""
    import soundfile as sf
    from asr_client import ASRWorkerExited

    bad_path = world["dir"] / "bench_clips" / "live" / "bad.wav"
    sf.write(str(bad_path), np.zeros(16000 * 4, dtype=np.float32), 16000)
    bad = {"path": str(bad_path), "category": "vtuber", "minutes": 4 / 60}

    class DiesOnBad(FakeASR):
        def transcribe(self, audio):
            if float(np.mean(audio)) > 0.4:
                raise ASRWorkerExited("worker exited")
            return super().transcribe(audio)

    def load(path):
        return np.full(16000, 0.5, dtype=np.float32) if str(path).endswith("bad.wav") else _load_audio(path)

    deps = _deps(world)
    deps.asr_client_factory = lambda cfg: DiesOnBad(cfg, world["by_id"])
    deps.vad_factory = _Vad
    deps.load_audio = load
    store = _run(RunStore.create(_plan(world, stages=[1, 2], recordings=[bad, recording])), deps)
    good = [r for r in store.read("vad_raw") if r["recording"] == recording["path"]]
    assert {r["asr"] for r in good} == set(store.frozen("top2")["vtuber"])
    for asr in store.frozen("top2")["vtuber"]:
        task = store.task(f"s2asr/{asr}")
        assert task["status"] == "failed" and "bad.wav" in task["reason"] and "NoneType" not in task["reason"]


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


# ---------------------------------------------------------------- worker leak

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


def _counting_asr(world, fail):
    """FakeASR whose transcribe call number n raises fail(n) (None: answer normally).
    Calls 1-3 are the warm-up, 4 the first sentence. Tracks live workers."""
    stats = {"calls": 0, "starts": 0, "live": 0, "max_live": 0}

    class Counting(FakeASR):
        def start(self):
            stats["starts"] += 1
            stats["live"] += 1
            stats["max_live"] = max(stats["max_live"], stats["live"])

        def transcribe(self, audio):
            stats["calls"] += 1
            error = fail(stats["calls"])
            if error is not None:
                raise error
            return super().transcribe(audio)

        def shutdown(self):
            stats["live"] -= 1

    deps = _deps(world)
    deps.asr_client_factory = lambda cfg: Counting(cfg, world["by_id"])
    return deps, stats


@pytest.mark.parametrize("failure", ["exited twice", "broken pipe"])
def test_sentence_the_worker_died_on_is_recorded_and_the_model_still_ranked(world, failure):
    """Review finding: when the worker died on a sentence (twice in a row, or with an
    error the session does not restart for) nothing was recorded. The ASR then had only
    n-1 of n sentences, was left out of the rankings, and a resume never retried it."""
    from asr_client import ASRWorkerExited

    def fail(n):
        if failure == "exited twice" and n in (5, 6):  # 6: the session's retry on a new worker
            return ASRWorkerExited("worker exited")
        if failure == "broken pipe" and n == 5:
            return BrokenPipeError("pipe closed")
        return None

    deps, stats = _counting_asr(world, fail)
    store = _run(RunStore.create(_plan(world, asr=["sensevoice"], stages=[1])), deps)
    assert {r["key"] for r in store.read("asr")} == {f"{ds}/{i:06d}" for ds, n in world["counts"].items()
                                                     for i in range(3, 3 + n)}
    errors = [r for r in store.read("asr") if r.get("error")]
    assert len(errors) == 1 and errors[0]["key"] == "anime/000004"
    assert stats["max_live"] == 1 and stats["live"] == 0
    assert analysis.asr_table(store)["anime"]["sensevoice"]["complete"] is True
    assert analysis.asr_rankings(store)["anime"] == ["sensevoice"]


def test_scattered_asr_errors_keep_the_model_but_errors_in_a_row_stop_it(world):
    """MAX_WORKER_ERRORS counts errors in a row (like FAIL_STREAK and main.py): three
    scattered single-sentence errors out of ten must not drop the whole ASR."""
    from asr_client import ASRWorkerError

    def failing_on(calls):
        return lambda n: ASRWorkerError({"message": "bad input"}) if n in calls else None

    deps, _stats = _counting_asr(world, failing_on({5, 7, 9}))
    store = _run(RunStore.create(_plan(world, asr=["sensevoice"], stages=[1])), deps)
    assert store.task("s1/sensevoice")["status"] == "done"
    assert analysis.asr_table(store)["anime"]["sensevoice"]["complete"] is True

    deps, _stats = _counting_asr(world, failing_on({5, 6, 7}))
    store = _run(RunStore.create(_plan(world, asr=["sensevoice"], stages=[1])), deps)
    assert store.task("s1/sensevoice")["status"] == "failed"
    assert analysis.asr_table(store)["anime"]["sensevoice"]["complete"] is False


def test_unrecoverable_worker_error_starts_a_fresh_worker(world):
    from asr_client import ASRWorkerError

    deps, stats = _counting_asr(
        world, lambda n: ASRWorkerError({"message": "CUDA context lost", "recoverable": False}) if n == 5 else None)
    store = _run(RunStore.create(_plan(world, asr=["sensevoice"], stages=[1])), deps)
    assert stats["starts"] == 2 and stats["max_live"] == 1 and stats["live"] == 0
    assert [r["key"] for r in store.read("asr") if r.get("error")] == ["anime/000004"]
    assert store.task("s1/sensevoice")["status"] == "done"


# ---------------------------------------------------------------- too few clips

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


def test_reusing_stage1_requires_the_same_clips_and_language(world):
    """Review finding: reuse compared only dataset names and counts. After the sample was
    drawn again, or with another forced language, stage 3 silently translated nothing
    (or text recognised under other settings)."""
    from bench import datasets

    ids = {d: datasets.sample_id(d, n) for d, n in world["counts"].items()}

    def old_run(**kw):
        run = RunStore.create(_plan(world, **kw))
        for c in ["sensevoice", "mlt-nano", "anime-whisper", "whisper-turbo"]:
            run.set_task(f"s1/{c}", status="done")
        return run.dir.name

    chosen, counts = ["anime", "vtuber"], world["counts"]
    old_run()  # a run from before data_ids were recorded
    assert cli._reusable_stage1(chosen, counts, ids, "ja") is None
    old_run(data_ids={**ids, "anime": {**ids["anime"], "clips": "another sample"}},
            settings_snapshot={"asr_language": "ja"})
    assert cli._reusable_stage1(chosen, counts, ids, "ja") is None
    good = old_run(data_ids=ids, settings_snapshot={"asr_language": "ja"})
    assert cli._reusable_stage1(chosen, counts, ids, "ja") == good
    assert cli._reusable_stage1(chosen, counts, ids, "auto") is None


def test_stage3_without_matching_stage1_records_fails_with_the_reason(world):
    old = _run(RunStore.create(_plan(world, stages=[1], asr=["sensevoice"])), _deps(world))
    for f in (world["dir"] / "bench_clips").glob("*/manifest.json"):  # the sample was drawn again
        m = json.loads(f.read_text(encoding="utf-8"))
        for c in m["clips"]:
            c["id"] = "9" + c["id"][1:]
            (f.parent / f"{c['id']}.wav").write_bytes((f.parent / f"0{c['id'][1:]}.wav").read_bytes())
        f.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    store = _run(RunStore.create(_one_model_plan(world, stages=[3], reuse_stage1_from=old.dir.name)), _deps(world))
    task = store.task(f"s3tl/{FP['A Model']}")
    assert task["status"] == "failed" and "對不上" in task["reason"]
    assert not store.read("tl")


def test_run_stops_when_test_clips_are_missing(world):
    (world["dir"] / "bench_clips" / "anime" / "000004.wav").unlink()
    store = RunStore.create(_plan(world, asr=["sensevoice"], stages=[1]))
    with pytest.raises(RuntimeError, match="3/4"):
        Bench(store, SETTINGS, CONFIG, _deps(world)).run()
    assert RunStore(store.dir).state["status"] == "error"
    assert not store.read("asr")


def test_resume_keeps_the_language_the_run_started_with(world, monkeypatch, tmp_path):
    RunStore.create(_plan(world, settings_snapshot={"asr_language": "ja"}))
    settings_path = tmp_path / "user_settings.json"
    settings_path.write_text(json.dumps({**SETTINGS, "asr_language": "auto"}), encoding="utf-8")
    monkeypatch.setattr(cli.settings_io.paths, "SETTINGS_FILE", settings_path)
    monkeypatch.setattr(settings_io, "find_running_app", lambda: [])
    used = {}
    monkeypatch.setattr(cli, "_run", lambda con, s, settings, config: used.update(settings) or 0)
    out = []
    assert cli.cmd_resume(cli.Console(input_fn=lambda p: "", print_fn=out.append)) == 0
    assert used["asr_language"] == "ja" and any("ja" in line and "auto" in line for line in out)


def test_cancelled_resume_keeps_the_failure_reasons(world, monkeypatch):
    """Review finding: failed tasks were reset before the final "start?" question, so
    answering n erased their reasons from the state and the report."""
    store = RunStore.create(_plan(world))
    store.set_task("s1/whisper-turbo", status="failed", reason="CUDA out of memory")
    store.set_task(f"s3judge/{FP['A Model']}", status="failed", reason="401")
    monkeypatch.setattr(settings_io, "find_running_app", lambda: [])
    monkeypatch.setattr(settings_io, "load_settings", lambda: SETTINGS)
    answers = iter(["y", "n"])  # retry the failed items: yes; start: no
    assert cli.cmd_resume(cli.Console(input_fn=lambda p: next(answers), print_fn=lambda s: None)) == 0
    again = RunStore(store.dir)
    assert again.task("s1/whisper-turbo") == {**again.task("s1/whisper-turbo"), "status": "failed",
                                              "reason": "CUDA out of memory"}
    assert again.task(f"s3judge/{FP['A Model']}")["reason"] == "401"


def test_reusing_stage1_requires_a_complete_matching_run(world, monkeypatch):
    old = RunStore.create(_plan(world, asr=["sensevoice"]))
    old.set_task("s1/sensevoice", status="done")
    monkeypatch.setattr(cli, "_model_options", lambda s: [(m, True, "已下載") for m in MODELS])
    answers = iter(["1", "", ""])  # untick stage 1 and confirm, datasets: confirm
    out = []
    plan = cli.build_plan(cli.Console(input_fn=lambda p: next(answers), print_fn=out.append), SETTINGS)
    assert plan is None and any("沒有完整的第 1 階段結果" in line for line in out)


# ---------------------------------------------------------------- fatal errors

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
    code = cli._run(cli.Console(input_fn=lambda p: "", print_fn=out.append), store, SETTINGS, CONFIG,
                    deps=_deps(world))
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


# ---------------------------------------------------------------- retries

def test_failed_sentences_of_an_interrupted_task_are_retried_on_resume(world):
    """The way a user gets there: some sentences fail, the window is closed mid-task,
    then "continue" runs the task again and redoes only what is not final."""
    class FailsThenStops(FakeTranslator):
        def translate_iter(self, text, source_language="ja"):
            self.calls += 1
            if self.calls in (4, 5):  # after the warm-up (3): two sentences fail
                raise TimeoutError("slow")
            if self.calls == 6:
                raise KeyboardInterrupt  # the window is closed
            yield f"{self.prefix}譯：{len(text)}字"

    store = RunStore.create(_one_model_plan(world))
    with pytest.raises(KeyboardInterrupt):
        Bench(store, SETTINGS, CONFIG, _deps(world, translator=lambda e, s, c: FailsThenStops(e))).run()
    store = RunStore(store.dir)
    assert store.state["status"] == "interrupted"
    assert [r["ok"] for r in store.read("tl")] == [False, False]
    _as_cmd_resume(store)
    store = _run(store, _deps(world))
    latest = {(r["category"], r["clip"]): r for r in store.read("tl")}
    assert len(latest) == 8 and all(r["ok"] for r in latest.values())
    assert analysis.quality_table(store)[(FP["A Model"], "anime")]["quality"] > 5


def test_judge_items_that_failed_are_scored_again_on_resume(world):
    class FirstAnswersFail(FakeJudge):
        def __init__(self, entry):
            super().__init__(entry)
            self.calls = 0

        def score(self, category, ja, tl):
            self.calls += 1
            return None if self.calls <= 3 else super().score(category, ja, tl)

    deps = _deps(world)
    deps.judge_factory = FirstAnswersFail
    store = _run(RunStore.create(_plan(world)), deps)
    assert sum(r["scores"] is None for r in store.read("judge")) == 2 * 3
    _as_cmd_resume(store)
    store = _run(store, _deps(world))
    latest = analysis._latest(store.read("judge"), ("judge", "item"))
    assert latest and all(r["scores"] for r in latest)
    assert analysis.quality_table(store)[(FP["A Model"], "anime")]["note"] is None


def test_translation_model_failing_in_a_row_is_stopped(world, monkeypatch):
    from bench import orchestrator

    monkeypatch.setattr(orchestrator, "FAIL_STREAK", 3)

    class DiesLater(FakeTranslator):
        def translate_iter(self, text, source_language="ja"):
            if self.calls >= 5:  # warm-up (3) + 2 good sentences, then the server is gone
                self.calls += 1
                raise ConnectionError("refused")
            yield from super().translate_iter(text, source_language)

    store = _run(RunStore.create(_one_model_plan(world)), _deps(world, translator=lambda e, s, c: DiesLater(e)))
    task = store.task(f"s3tl/{FP['A Model']}")
    assert task["status"] == "failed" and "連續 3 句失敗" in task["reason"] and "refused" in task["reason"]
    assert [r["ok"] for r in store.read("tl")] == [True, True, False, False, False]
    assert {m["reason"] for m in store.read("together_meta")} == {"翻譯模型無法使用"}


def test_judge_failing_every_calibration_item_is_stopped(world):
    class NeverScores(FakeJudge):
        def __init__(self, entry):
            super().__init__(entry)
            self.last_error = "BadRequestError: seed is not supported"

        def score(self, category, ja, tl):
            return None

    deps = _deps(world)
    deps.judge_factory = NeverScores
    store = _run(RunStore.create(_plan(world)), deps)
    for jfp in (FP["A Model"], FP["B Model"]):
        task = store.task(f"s3judge/{jfp}")
        assert task["status"] == "failed" and "校準題全部無法評分" in task["reason"]
        assert "seed is not supported" in task["reason"]
        assert sum(r["judge"] == jfp for r in store.read("judge")) == 6


def test_rankings_stay_frozen_when_stage1_is_retried(world):
    """Later stages used the first rankings; a retried stage-1 task updates the stage-1
    table only, so the run stays consistent."""
    class SenseVoiceFails(FakeASR):
        def start(self):
            if self.config.get("funasr_model") == "sensevoice-small":
                raise RuntimeError("no GPU memory")

    deps = _deps(world)
    deps.asr_client_factory = lambda cfg: SenseVoiceFails(cfg, world["by_id"])
    store = _run(RunStore.create(_plan(world, asr=["sensevoice", "mlt-nano", "anime-whisper"], stages=[1])), deps)
    first = store.frozen("rankings")
    assert "sensevoice" not in first["anime"]
    store.set_task("s1/sensevoice", status="pending")
    store = _run(store, _deps(world))
    assert store.frozen("rankings") == first
    assert analysis.asr_rankings(store)["anime"][0] == "sensevoice"


# ---------------------------------------------------------------- stage 1 meta

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


# ---------------------------------------------------------------- failures under load

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


class _FailsUnderLoad(FakeTranslator):
    """Translations on the replay pool fail (the warm-up on the main thread works)."""

    def translate_iter(self, text, source_language="ja"):
        import threading

        if threading.current_thread().name.startswith("bench-tl"):
            raise TimeoutError("busy GPU")
        yield from super().translate_iter(text, source_language)


def _one_model_plan(world, **kw):
    return _plan(world, tl_models=_plan(world)["tl_models"][:1], judges=[FP["A Model"]], **kw)


def test_failed_together_sentences_are_not_remeasured_alone_on_resume(world):
    """Review finding: a resume replayed only the sentences that had failed under load,
    now without the queue in front of them, and the easy success replaced the failure."""
    store = _run(RunStore.create(_one_model_plan(world)), _deps(world, translator=lambda e, s, c: _FailsUnderLoad(e)))
    before = store.read("together")
    assert before and not any(r["ok"] for r in before)
    store.set_task(before[0]["task"], status="pending")  # as if stopped during its last category
    store.state["status"] = "running"
    store.save()
    store = _run(store, _deps(world))
    assert len(store.read("together")) == len(before)


def test_together_failures_keep_their_reason(world, caplog):
    deps = _deps(world, translator=lambda e, s, c: _FailsUnderLoad(e))
    with caplog.at_level("WARNING"):
        store = _run(RunStore.create(_one_model_plan(world)), deps)
    together = store.read("together")
    assert together and all(r["error"] == "timeout" and "busy GPU" in r["detail"] for r in together)
    assert any("busy GPU" in r.message for r in caplog.records if "Together" in r.message)
    meta = [m for m in store.read("together_meta") if m["status"] == "ok"]
    assert meta and all(m["errors"] == {"timeout": sum(r["task"] == m["task"] for r in together)} for m in meta)
    assert "失敗：timeout 4 句" in deps.out.getvalue()  # SenseVoice: 2 categories x 2 sentences


def test_pair_whose_warm_up_translations_all_fail_is_marked_failed(world):
    class NoMemory:
        def translate_iter(self, text, source_language="ja"):
            raise RuntimeError("model requires more system memory")
            yield  # pragma: no cover

    made = {"n": 0}

    def factory(entry, s, c):  # the model works alone (stage 3 translation), not next to the ASR
        made["n"] += 1
        return FakeTranslator(entry) if made["n"] == 1 else NoMemory()

    store = _run(RunStore.create(_one_model_plan(world)), _deps(world, translator=factory))
    tog = [t for t in store.state["tasks"] if t["id"].startswith("s3tog/")]
    assert tog and all(t["status"] == "failed" and "requires more system memory" in t["reason"] for t in tog)
    assert not store.read("together")


# ---------------------------------------------------------------- unfinished runs

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


def test_recommendation_written_by_a_run_is_applied_as_written(world, monkeypatch, tmp_path):
    """G7: recommendations.json from a real run read back by the apply menu (a renamed
    field on either side used to pass every test)."""
    import local_models

    store = _run(RunStore.create(_plan(world)), _deps(world))
    first = json.loads((store.dir / "recommendations.json").read_text(encoding="utf-8"))["items"][0]
    settings_path = tmp_path / "user_settings.json"
    settings_path.write_text(json.dumps({**SETTINGS, "asr_engine": "whisper", "vad_threshold": 0.45,
                                         "active_model": 2}), encoding="utf-8")
    monkeypatch.setattr(cli.settings_io.paths, "SETTINGS_FILE", settings_path)
    monkeypatch.setattr(local_models, "SELECTED_MODEL_FILE", tmp_path / "selected_model.txt")
    monkeypatch.setattr(settings_io, "find_running_app", lambda: [])
    monkeypatch.setattr(cli, "is_cached", lambda cand, hub: True)
    answers = iter(["", "", "y"])  # newest run, first recommendation, confirm
    assert cli.cmd_apply(cli.Console(input_fn=lambda p: next(answers), print_fn=lambda s: None)) == 0
    after = json.loads(settings_path.read_text(encoding="utf-8"))
    from bench.asr_runner import CANDIDATES_BY_ID

    assert {k: after[k] for k in CANDIDATES_BY_ID[first["asr"]].app_settings()} == \
        CANDIDATES_BY_ID[first["asr"]].app_settings()
    assert after["active_model"] == [m["name"] for m in MODELS].index(first["tl_name"])
    assert after["vad_threshold"] == 0.45  # no recordings: not measured, kept


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


# ---------------------------------------------------------------- apply that would be undone

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
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["active_model"] == 0 and after["asr_engine"] == "funasr"
    assert not (tmp_path / "selected_model.txt").exists()  # a cloud model is not the launcher's choice


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


# ---------------------------------------------------------------- failures mid-run

def test_model_that_breaks_mid_run_reports_its_errors(world):
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

    class Session:  # long enough to tell "from the audio's end" from "from the translation's start"
        def transcribe(self, audio):
            time.sleep(0.15)
            return {"text": "本当にそう思う", "language": "ja"}, 150.0

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
        # ASR 0.15 s + translation 0.1 s, counted from when the sentence was handed over;
        # well below the whole replay (~1.4 s) even on a busy machine
        assert 0.24 <= r["e2e_ms"] / 1000 < 0.9, r["e2e_ms"]
