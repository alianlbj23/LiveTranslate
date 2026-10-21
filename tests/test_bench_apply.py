"""Applying a recommended combination: backup first, only the needed keys, the launcher's
selected_model.txt, refusal while LiveTranslate runs, restore. All paths are sandboxed
(tests/conftest.py points bench.paths and local_models at a temp folder)."""
import json

import pytest

import local_models
from bench import cli, paths, settings_io, state
from bench.wizard import Console

MODELS = [
    {"name": "Qwen3.5 9B", "model": "qwen3.5:9b", "api_base": "http://127.0.0.1:11434/v1", "api_key": "ollama"},
    {"name": "Qwen3.5 4B", "model": "qwen3.5:4b", "api_base": "http://127.0.0.1:11434/v1", "api_key": "ollama"},
    {"name": "Cloud", "model": "gpt-x", "api_base": "https://api.example.com/v1", "api_key": "k"},
]


@pytest.fixture
def user_files(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVETRANSLATE_BENCH_DIR", str(tmp_path / "bench"))
    settings = {"asr_engine": "whisper", "funasr_model": "sensevoice-small", "vad_threshold": 0.5,
                "active_model": 0, "models": MODELS, "style": {"bg_color": "#000000"}, "overlay_x": 10}
    path = tmp_path / "user_settings.json"
    path.write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8")
    selected = tmp_path / "selected_model.txt"
    selected.write_text("qwen3.5:9b", encoding="utf-8")
    monkeypatch.setattr(paths, "SETTINGS_FILE", path)
    monkeypatch.setattr(local_models, "SELECTED_MODEL_FILE", selected)
    monkeypatch.setattr(settings_io, "find_running_app", lambda: [])
    return {"settings": path, "selected": selected, "original": settings}


def test_planned_changes_only_lists_what_differs(user_files):
    s = settings_io.load_settings()
    changes = settings_io.planned_changes(s, "sensevoice", 0.4, MODELS[1])
    assert changes == {"asr_engine": ("whisper", "funasr"), "vad_threshold": (0.5, 0.4), "active_model": (0, 1)}


def test_whisper_turbo_needs_the_local_model(user_files, monkeypatch):
    import bench.asr_runner as asr_runner

    monkeypatch.setattr(asr_runner, "find_turbo_path", lambda: None)
    with pytest.raises(settings_io.SettingsError, match="turbo"):
        settings_io.planned_changes(settings_io.load_settings(), "whisper-turbo", 0.5, MODELS[0])
    monkeypatch.setattr(asr_runner, "find_turbo_path", lambda: r"D:\models\turbo")
    changes = settings_io.planned_changes(settings_io.load_settings(), "whisper-turbo", 0.5, MODELS[0])
    assert changes == {"whisper_model_size": (None, r"D:\models\turbo")}


def test_apply_backs_up_writes_only_the_keys_and_the_selected_model(user_files):
    s = settings_io.load_settings()
    changes = settings_io.planned_changes(s, "mlt-nano", 0.4, MODELS[1])
    saved = settings_io.apply_changes(changes, MODELS[1])
    after = json.loads(user_files["settings"].read_text(encoding="utf-8"))
    assert after["asr_engine"] == "funasr" and after["funasr_model"] == "funasr-mlt-nano-2512"
    assert after["vad_threshold"] == 0.4 and after["active_model"] == 1
    untouched = {k: v for k, v in after.items() if k not in changes}
    assert untouched == {k: v for k, v in user_files["original"].items() if k not in changes}
    assert json.loads((saved / "user_settings.json").read_text(encoding="utf-8")) == user_files["original"]
    assert (saved / "selected_model.txt").read_text(encoding="utf-8") == "qwen3.5:9b"
    assert user_files["selected"].read_text(encoding="utf-8") == "qwen3.5:4b"

    settings_io.restore(saved)
    assert json.loads(user_files["settings"].read_text(encoding="utf-8")) == user_files["original"]
    assert user_files["selected"].read_text(encoding="utf-8") == "qwen3.5:9b"


def test_cloud_model_does_not_touch_the_launcher_selection(user_files):
    changes = settings_io.planned_changes(settings_io.load_settings(), "sensevoice", 0.5, MODELS[2])
    settings_io.apply_changes(changes, MODELS[2])
    assert user_files["selected"].read_text(encoding="utf-8") == "qwen3.5:9b"


def test_apply_refuses_while_livetranslate_runs(user_files, monkeypatch):
    monkeypatch.setattr(settings_io, "find_running_app", lambda: [1234])
    changes = settings_io.planned_changes(settings_io.load_settings(), "sensevoice", 0.4, MODELS[1])
    with pytest.raises(settings_io.SettingsError, match="正在執行"):
        settings_io.apply_changes(changes, MODELS[1])
    assert json.loads(user_files["settings"].read_text(encoding="utf-8")) == user_files["original"]


def test_unknown_translation_model_is_an_error(user_files):
    with pytest.raises(settings_io.SettingsError, match="找不到翻譯模型"):
        settings_io.planned_changes(settings_io.load_settings(), "sensevoice", 0.4, {"name": "X", "model": "y"})


def test_broken_settings_file_is_reported(tmp_path):
    bad = tmp_path / "s.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(settings_io.SettingsError, match="格式錯誤"):
        settings_io.load_settings(bad)
    with pytest.raises(settings_io.SettingsError, match="找不到設定檔"):
        settings_io.load_settings(tmp_path / "missing.json")


def test_setup_steps_use_the_control_panel_labels():
    steps = settings_io.setup_steps("sensevoice", 0.4, "Qwen3.5 9B", "A1")
    labels = settings_io.ui_labels()
    assert labels["label_engine"] in steps[0] and "FunASR" in steps[0] and "SenseVoice" in steps[0]
    assert "0.40" in steps[1] and labels["group_silero_threshold"] in steps[1]
    assert "Qwen3.5 9B" in steps[2] and labels["group_active_model"] in steps[2]
    assert "A1" in steps[3]
    assert "Anime-Whisper" in settings_io.setup_steps("anime-whisper", 0.5, "X", "V2")[0]


def _pick(asr="sensevoice", model=MODELS[1], threshold=0.4, measured=True, pick_id="A1"):
    """A recommendation as analysis.recommendations returns it (report.rec_item's input)."""
    from bench.llm_servers import fingerprint

    return {"id": pick_id, "kinds": ["品質最好"], "combo": {
        "asr": asr, "vad_threshold": threshold, "vad_measured": measured, "tl": fingerprint(model),
        "tl_name": model["name"], "tl_model": model["model"], "composite": 80.0, "quality": 8.0,
        "latency_ms": 900.0, "recommended": True}}


def _run_with_recs(*picks, status="done", raw_items=None):
    from bench import report

    run = state.RunStore.create({"stages": [1]})
    run.set_status(status)
    items = raw_items if raw_items is not None else [report.rec_item("anime", p) for p in picks]
    (run.dir / "recommendations.json").write_text(json.dumps(
        {"version": report.RECS_VERSION, "run": run.dir.name, "status": status, "items": items},
        ensure_ascii=False), encoding="utf-8")
    return run


def _apply(*answers):
    answers = iter(answers)
    out = []
    code = cli.cmd_apply(Console(input_fn=lambda p: next(answers), print_fn=out.append))
    return code, out


def test_apply_menu_flow(user_files):
    _run_with_recs(_pick())
    code, out = _apply("", "", "y")  # newest run, A1, confirm
    assert code == 0
    after = json.loads(user_files["settings"].read_text(encoding="utf-8"))
    assert after["asr_engine"] == "funasr" and after["active_model"] == 1
    assert any("已套用" in line for line in out) and any("asr_engine" in line for line in out)


def test_unmeasured_vad_threshold_keeps_the_current_one(user_files):
    """G8: a recommendation without its own recordings must not overwrite the user's
    threshold with the default."""
    settings = json.loads(user_files["settings"].read_text(encoding="utf-8"))
    user_files["settings"].write_text(json.dumps({**settings, "vad_threshold": 0.35}), encoding="utf-8")
    _run_with_recs(_pick(threshold=0.5, measured=False))
    code, out = _apply("", "", "y")
    assert code == 0 and json.loads(user_files["settings"].read_text(encoding="utf-8"))["vad_threshold"] == 0.35
    assert any("沒有實測" in line for line in out) and any("未實測" in line for line in out)


def test_recommendation_that_cannot_be_applied_is_left_out_not_a_traceback(user_files, caplog):
    """Review finding: an unknown ASR id or a renamed field in recommendations.json ended
    in a KeyError traceback inside the menu."""
    from bench import report

    good = report.rec_item("anime", _pick(pick_id="A2"))
    bad_asr = {**report.rec_item("anime", _pick()), "asr": "whisper-large-v4"}
    renamed = {k: v for k, v in report.rec_item("anime", _pick(pick_id="A3")).items() if k != "vad_measured"}
    run = _run_with_recs(raw_items=[bad_asr, renamed, good])
    with caplog.at_level("WARNING"):
        assert [r["id"] for r in report.load_recs(run.dir)["items"]] == ["A2"]
    assert any("whisper-large-v4" in r.message for r in caplog.records)
    assert any("vad_measured" in r.message for r in caplog.records)
    code, _out = _apply("", "", "y")
    assert code == 0 and json.loads(user_files["settings"].read_text(encoding="utf-8"))["asr_engine"] == "funasr"


def test_unknown_asr_is_a_settings_error(user_files):
    with pytest.raises(settings_io.SettingsError, match="不認得"):
        settings_io.planned_changes(settings_io.load_settings(), "whisper-large-v4", 0.4, MODELS[0])
    with pytest.raises(settings_io.SettingsError, match="不認得"):
        settings_io.setup_steps("whisper-large-v4", 0.4, "X", "A1")


def test_restoring_twice_still_returns_to_the_settings_before_the_apply(user_files):
    """Review finding: the first restore backed up the applied settings, and the second
    "restore the settings from before the last apply" picked that backup instead."""
    _run_with_recs(_pick())
    assert _apply("", "", "y")[0] == 0
    for _ in range(2):
        code, out = _apply("", "2", "y")  # newest run, R (after A1), confirm
        assert code == 0 and any("已還原" in line for line in out)
        assert json.loads(user_files["settings"].read_text(encoding="utf-8")) == user_files["original"]
    assert len(settings_io.list_backups(include_pre_restore=True)) == 3


def test_cancelling_the_run_choice_leaves_the_menu(user_files):
    """Review finding: 0 at "choose a run" looked like "no runs" and, with backups,
    opened a menu holding only the restore."""
    _run_with_recs(_pick())
    assert _apply("", "", "y")[0] == 0  # leaves a backup behind
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return "0"

    assert cli.cmd_apply(Console(input_fn=answer, print_fn=lambda s: None)) == 0
    assert len(prompts) == 1


def test_restore_works_when_the_settings_file_is_broken(user_files):
    """Review finding: the menu read the settings before offering R, so a broken file
    made the restore unreachable."""
    _run_with_recs(_pick())
    assert _apply("", "", "y")[0] == 0
    user_files["settings"].write_text("{broken", encoding="utf-8")
    code, out = _apply("", "2", "y")
    assert code == 0 and any("已還原" in line for line in out)
    assert json.loads(user_files["settings"].read_text(encoding="utf-8")) == user_files["original"]


def test_broken_config_yaml_is_a_settings_error(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("asr: [unclosed", encoding="utf-8")
    with pytest.raises(settings_io.SettingsError, match="格式錯誤"):
        settings_io.load_config(bad)
    with pytest.raises(settings_io.SettingsError, match="讀不到"):
        settings_io.load_config(tmp_path / "missing.yaml")
