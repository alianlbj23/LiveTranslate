"""Pure logic of the benchmark: VRAM settle, run state, scoring, judging, costs, menus."""
import json
import os

import pytest

from bench import costs, gpu, judge, scoring, state
from bench.progress import Ewma, ProgressLine, fmt_duration
from bench.wizard import Console


@pytest.fixture
def bench_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVETRANSLATE_BENCH_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------- VRAM

class _Reader:
    def __init__(self, values):
        self.values = list(values)

    def used_mb(self):
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_vram_settles_when_back_near_the_baseline():
    clock = _Clock()
    r = gpu.wait_vram_settle(_Reader([9000, 6000, 2200]), 2000, clock=clock, sleep=clock.sleep)
    assert r["status"] == "settled" and r["used_mb"] == 2200 and r["waited_s"] == 1.0


def test_vram_stable_but_high_is_accepted_after_a_while():
    clock = _Clock()
    r = gpu.wait_vram_settle(_Reader([2800]), 2000, clock=clock, sleep=clock.sleep)
    assert r["status"] == "stable_high" and 15 <= r["waited_s"] <= 16


def test_vram_timeout_when_memory_never_falls():
    clock = _Clock()
    r = gpu.wait_vram_settle(_Reader([9000]), 2000, timeout=20, clock=clock, sleep=clock.sleep)
    assert r["status"] == "timeout"


def test_vram_without_nvml_waits_a_fixed_time():
    clock = _Clock()
    r = gpu.wait_vram_settle(_Reader([None]), None, clock=clock, sleep=clock.sleep)
    assert r["status"] == "no_nvml" and clock.t == 10.0


def test_vram_sampler_keeps_the_peak():
    with gpu.VramSampler(_Reader([1000, 5000, 3000]), interval=0.001) as s:
        pass
    assert s.peak_mb == 5000


# ---------------------------------------------------------------- state

def test_run_store_appends_and_survives_a_cut_off_line(bench_dir):
    store = state.RunStore.create({"stages": [1]})
    store.append("asr", {"task": "s1/x", "key": "a/1"})
    store.append("asr", {"task": "s1/x", "key": "a/2"})
    with open(store.dir / "results" / "asr.jsonl", "a", encoding="utf-8") as f:
        f.write('{"task": "s1/x", "key": "a/')  # crash mid-write
    reopened = state.RunStore(store.dir)
    assert reopened.done_keys("asr", "s1/x") == {"a/1", "a/2"}


def test_run_store_tasks_and_frozen_values_persist(bench_dir):
    store = state.RunStore.create({"stages": [1]})
    store.set_task("s1/x", status="done", items_done=3)
    store.freeze("rankings", {"anime": ["a"]})
    again = state.RunStore(store.dir)
    assert again.task("s1/x")["status"] == "done"
    assert again.frozen("rankings") == {"anime": ["a"]}
    assert state.RunStore.latest_unfinished().dir == store.dir
    again.state["status"] = "done"
    again.save()
    assert state.RunStore.latest_unfinished() is None


def test_run_folders_never_collide(bench_dir):
    from datetime import datetime

    now = datetime(2026, 10, 20, 21, 30)
    a = state.RunStore.create({}, now=now)
    b = state.RunStore.create({}, now=now)
    assert a.dir != b.dir


def test_stale_lock_is_ignored(bench_dir, monkeypatch):
    import psutil

    state.lock_path().parent.mkdir(parents=True)
    state.lock_path().write_text("999999", encoding="ascii")
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: False)
    assert state.running_pid() is None
    assert state.acquire_lock()
    assert state.lock_path().read_text(encoding="ascii") == str(os.getpid())
    state.release_lock()
    assert not state.lock_path().exists()


def test_live_lock_blocks_a_second_run(bench_dir, monkeypatch):
    import psutil

    state.lock_path().parent.mkdir(parents=True)
    state.lock_path().write_text("4242", encoding="ascii")
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: pid == 4242)
    assert state.running_pid() == 4242
    assert not state.acquire_lock()


# ---------------------------------------------------------------- progress

@pytest.mark.parametrize("seconds, text", [(40, "40 秒"), (125, "2 分"), (4320, "1 小時 12 分"), (-5, "0 秒")])
def test_fmt_duration(seconds, text):
    assert fmt_duration(seconds) == text


def test_ewma_and_progress_line():
    e = Ewma(alpha=0.5)
    assert e.add(100) == 100 and e.add(200) == 150
    import io

    out = io.StringIO()
    clock = _Clock()
    line = ProgressLine(out=out, clock=clock)
    line.show("階段 1/3 辨識", "anime", 1, 10, 120.0, 3600)
    line.show("階段 1/3 辨識", "anime", 2, 10, 120.0, 3600)  # throttled
    clock.t = 2.0
    line.show("階段 1/3 辨識", "anime", 3, 10, 120.0, 3600)
    text = out.getvalue()
    assert "第 1/10 句" in text and "第 2/10 句" not in text and "第 3/10 句" in text
    assert "預估剩餘 1 小時 0 分" in text


# ---------------------------------------------------------------- scoring

def test_composite_formula_and_latency_anchors():
    assert scoring.latency_points(500) == 1.0 and scoring.latency_points(3000) == 0.0
    assert scoring.latency_points(1750) == pytest.approx(0.5)
    # quality 10, CER 0, latency 0.5 s -> 100
    assert scoring.composite(10, 0.0, 500) == pytest.approx(100.0)
    # quality 1 -> q 0; CER 1 -> r 0; latency 3 s -> l 0
    assert scoring.composite(1, 1.0, 3000) == pytest.approx(0.0)
    assert scoring.composite(5.5, 0.1, 1750) == pytest.approx(100 * (0.6 * 0.5 + 0.2 * 0.9 + 0.2 * 0.5))


def _combo(**kw):
    base = {"quality": 7.0, "cer": 0.1, "latency_ms": 1000.0, "vram_peak_mb": 8000.0, "fail_rate": 0.0}
    base.update(kw)
    return scoring.judge_combo(base)


def test_not_recommended_rules():
    assert _combo(latency_ms=2000.0)["recommended"]
    assert not _combo(latency_ms=2001.0)["recommended"]
    assert _combo(vram_peak_mb=11 * 1024)["recommended"]
    assert not _combo(vram_peak_mb=11 * 1024 + 1)["recommended"]
    missing = _combo(quality=None)
    assert not missing["recommended"] and missing["composite"] is None
    warned = _combo(fail_rate=0.2, self_judged=True)
    assert warned["recommended"] and len(warned["warnings"]) == 2


def test_three_recommendations_merge_when_one_combo_wins_twice():
    a = _combo(quality=9.0, latency_ms=1500.0)
    b = _combo(quality=6.5, latency_ms=600.0)
    c = _combo(quality=5.0, latency_ms=400.0)  # fastest but below the quality floor
    picks = scoring.recommend([a, b, c], "A")
    kinds = {p["id"]: p["kinds"] for p in picks}
    assert picks[0]["combo"] is a and "品質最好" in picks[0]["kinds"]
    fastest = next(p for p in picks if "延遲最低" in p["kinds"])
    assert fastest["combo"] is b
    assert sum(len(k) for k in kinds.values()) == 3
    assert all(p["id"].startswith("A") for p in picks)


def test_recommendations_fall_back_with_a_warning():
    slow = _combo(latency_ms=2500.0)
    picks = scoring.recommend([slow], "V")
    assert picks and picks[0]["fallback"] and picks[0]["combo"] is slow


def test_rank_asr_skips_incomplete():
    stats = {"a": {"cer": 0.2, "mean_ms": 300, "complete": True}, "b": {"cer": 0.1, "mean_ms": 900, "complete": True},
             "c": {"cer": 0.05, "mean_ms": 100, "complete": False}, "d": {"cer": 0.1, "mean_ms": 200, "complete": True}}
    assert scoring.rank_asr(stats) == ["d", "b", "a"]


# ---------------------------------------------------------------- judge

@pytest.mark.parametrize("text, expected", [
    ('{"accuracy":8,"naturalness":7,"tone":6,"taiwan":9,"note":"ok"}', (8, 7, 6, 9)),
    ('```json\n{"accuracy": 7.6, "naturalness": "6", "tone": 10, "taiwan": 0}\n```', (8, 6, 10, 1)),
    ('評分如下：{"accuracy":5,"naturalness":5,"tone":5,"taiwan":5} 以上', (5, 5, 5, 5)),
])
def test_parse_scores_tolerates_noise(text, expected):
    s = judge.parse_scores(text)
    assert tuple(s[c] for c in judge.CRITERIA) == expected


@pytest.mark.parametrize("text", ["", "不知道", '{"accuracy":8}', '{"accuracy":"x","naturalness":1,"tone":1,"taiwan":1}',
                                  '{"accuracy":85,"naturalness":70,"tone":60,"taiwan":90}'])  # 100-point scale
def test_parse_scores_rejects_incomplete(text):
    assert judge.parse_scores(text) is None


def test_parse_vad_drops_out_of_range_lines():
    r = judge.parse_vad('{"hallucinated":[1,3,9,0],"missed":["お疲れ様","",5]}', n_lines=3)
    assert r == {"hallucinated": [1, 3], "missed": ["お疲れ様", "5"]}
    assert judge.parse_vad("nope", 3) is None


def test_cross_judging_never_scores_itself_when_another_judge_exists():
    a = judge.assign_judges(["A", "B", "S"], ["A", "B"])
    assert a["A"] == [("B", False)] and a["B"] == [("A", False)]
    assert a["S"] == [("A", False), ("B", False)]
    single = judge.assign_judges(["A", "S"], ["A"])
    assert single["A"] == [("A", True)] and single["S"] == [("A", False)]


def test_judge_offsets_align_scales_on_shared_items():
    def rec(j, item, v):
        return {"judge": j, "item": item, "scores": {c: v for c in judge.CRITERIA}}
    recs = [rec("A", "x", 8), rec("B", "x", 6), rec("A", "y", 9), rec("B", "y", 7), rec("A", "only-a", 3)]
    off = judge.judge_offsets(recs)
    assert off["A"] == pytest.approx(1.0) and off["B"] == pytest.approx(-1.0)
    assert judge.judge_offsets([rec("A", "x", 5)]) == {"A": 0.0}


def test_calibration_flags_an_unreliable_judge():
    def rec(j, cid, scores):
        return {"judge": j, "item": f"cal/{cid}", "scores": scores}
    good = {c: 9 for c in judge.CRITERIA}
    bad = {c: 9 for c in judge.CRITERIA}
    recs = [rec("A", "cal-untranslated", {**good, "accuracy": 2}), rec("A", "cal-opposite", {**good, "accuracy": 1}),
            rec("B", "cal-untranslated", bad), rec("B", "cal-opposite", bad), rec("B", "cal-good", None)]
    report = judge.calibration_report(recs)
    assert report["A"]["misses"] == 0
    assert report["B"]["misses"] == 3


def test_sakura_cannot_judge():
    assert not judge.eligible_judge({"prompt_style": "sakura"})
    assert judge.eligible_judge({"model": "qwen3.5:9b"})


class _FakeChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls.append(kw)
        text = self.replies.pop(0)
        if isinstance(text, Exception):
            raise text
        usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 20})()
        msg = type("M", (), {"content": text})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()], "usage": usage})()


def _entry(**kw):
    e = {"name": "Qwen3.5 9B", "model": "qwen3.5:9b", "api_base": "http://127.0.0.1:11434/v1", "api_key": "ollama",
         "thinking_style": "openai"}
    e.update(kw)
    return e


def test_judge_client_disables_thinking_and_retries_once():
    chat = _FakeChat(["好的", '{"accuracy":8,"naturalness":7,"tone":6,"taiwan":9}'])
    client = judge.JudgeClient(_entry(), client=chat)
    scores = client.score("anime", "ありがとう", "謝謝")
    assert scores["accuracy"] == 8
    assert len(chat.calls) == 2 and "只輸出 JSON" in chat.calls[1]["messages"][-1]["content"]
    assert chat.calls[0]["extra_body"] == {"reasoning_effort": "none"}
    assert chat.calls[0]["temperature"] == 0
    assert client.prompt_tokens == 200


def test_judge_client_gives_up_after_errors():
    chat = _FakeChat([RuntimeError("down"), RuntimeError("down")])
    assert judge.JudgeClient(_entry(), client=chat).score("vtuber", "a", "b") is None


def test_judge_prompt_is_blind():
    prompt = judge.tl_user_prompt("anime", "ありがとう", "謝謝")
    assert "Qwen" not in prompt and "qwen" not in judge.TL_SYSTEM.lower()


# ---------------------------------------------------------------- costs

def test_token_estimate_and_cost():
    assert costs.est_tokens("ありがとう") == 5
    assert costs.est_tokens("hello world!") == 3
    local = _entry()
    assert costs.call_cost(local, 1_000_000, 1_000_000) == 0.0
    cloud = {"name": "gpt", "model": "gpt-x", "api_base": "https://api.example.com/v1", "api_key": "k",
             "input_price": 1.0, "output_price": 4.0}
    assert costs.call_cost(cloud, 1_000_000, 500_000) == pytest.approx(3.0)
    assert costs.call_cost({**cloud, "input_price": 0, "output_price": 0}, 10, 10) is None
    assert costs.fmt_cost(None) == "未設定價格"
    assert costs.fmt_cost(0.0).startswith("US$0")
    assert costs.fmt_cost(1.0) == "約 US$0.70–1.30"


# ---------------------------------------------------------------- wizard

def _console(answers):
    out = []
    it = iter(answers)
    return Console(input_fn=lambda prompt: next(it), print_fn=out.append), out


def test_choose_many_toggles_and_confirms():
    con, out = _console(["3", "1", "", ])
    opts = [("a", "A", True, ""), ("b", "B", False, "沒有錄音"), ("c", "C", True, "")]
    assert con.choose_many("?", opts, {"a"}) == ["c"]


def test_choose_many_refuses_disabled_and_empty():
    con, out = _console(["2", "1", "", "1", ""])
    opts = [("a", "A", True, ""), ("b", "B", False, "沒有錄音")]
    assert con.choose_many("?", opts, {"a"}) == ["a"]
    assert any("不能勾選" in line for line in out) and any("至少要勾選" in line for line in out)


def test_choose_one_default_cancel_and_retry():
    con, out = _console(["", "0", "9", "2"])
    opts = [("x", "X"), ("y", "Y")]
    assert con.choose_one("?", opts, 0) == "x"
    assert con.choose_one("?", opts, 0) is None
    assert con.choose_one("?", opts, 0) == "y"


def test_confirm_and_end_of_input():
    con, _ = _console(["", "n", "maybe", "y"])
    assert con.confirm("?", True) is True
    assert con.confirm("?", True) is False
    assert con.confirm("?", False) is True

    def eof(prompt):
        raise EOFError
    assert Console(input_fn=eof, print_fn=lambda *_: None).choose_one("?", [("x", "X")]) is None


def test_plan_json_holds_no_api_key(bench_dir):
    store = state.RunStore.create({"tl_models": [{"fp": "Qwen|qwen|http://x", "name": "Qwen", "model": "qwen"}]})
    assert "api_key" not in json.loads((store.dir / "plan.json").read_text(encoding="utf-8"))["tl_models"][0]
