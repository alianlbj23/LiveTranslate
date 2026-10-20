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


def test_apply_menu_flow(user_files):
    run = state.RunStore.create({"stages": [1]})
    run.state["status"] = "done"
    run.save()
    (run.dir / "recommendations.json").write_text(json.dumps([
        {"id": "A1", "kinds": ["品質最好"], "category": "anime", "asr": "sensevoice", "asr_label": "SenseVoice Small",
         "vad_threshold": 0.4, "tl_name": "Qwen3.5 4B", "tl_model": "qwen3.5:4b", "composite": 80.0,
         "quality": 8.0, "latency_ms": 900.0, "recommended": True}], ensure_ascii=False), encoding="utf-8")
    answers = iter(["", "", "y"])  # newest run, A1, confirm
    out = []
    code = cli.cmd_apply(Console(input_fn=lambda p: next(answers), print_fn=out.append))
    assert code == 0
    after = json.loads(user_files["settings"].read_text(encoding="utf-8"))
    assert after["asr_engine"] == "funasr" and after["active_model"] == 1
    assert any("已套用" in line for line in out) and any("asr_engine" in line for line in out)
