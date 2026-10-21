"""Recommended configuration for this computer (hardware_profile.py, launcher/hardware_advice.py).

Users should not end up with a speech + translation model pair their graphics card cannot
hold: both run on the GPU together, and when memory runs out Ollama falls back to the
CPU and every subtitle takes 10 s or more. The GPU memory each model needs comes from
the best-combination benchmark (2026-10-20); judgments use the card's total memory."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import hardware_profile as hp

ROOT = Path(__file__).resolve().parents[1]
SAKURA = "hf.co/SakuraLLM/Sakura-GalTransl-7B-v3:IQ4_XS"
TURBO = r"D:\app\models\huggingface\hub\models--mobiuslabsgmbh--faster-whisper-large-v3-turbo\snapshots\abc"


def _gpu(total_gb, free_gb=None, name="NVIDIA GeForce RTX 5070"):
    return {"name": name, "total_gb": total_gb, "free_gb": total_gb if free_gb is None else free_gb}


# ---------------------------------------------------------------- catalog and tables

def test_every_catalog_model_says_how_much_gpu_memory_it_needs():
    catalog = hp.translation_models(ROOT)
    assert {m["model"] for m in catalog} >= {"qwen3.5:9b", "qwen3.5:4b", SAKURA}
    for m in catalog:
        assert isinstance(m["vram_gb"], (int, float)) and 1 <= m["vram_gb"] <= 24, m
        assert isinstance(m["rank"], int), m
    assert len({m["rank"] for m in catalog}) == len(catalog)


def test_bad_numbers_in_the_catalog_do_not_break_it(tmp_path, caplog):
    """Review finding: "size_gb": "6.6 GB" raised and the whole catalog (model list,
    launcher sync, advice) failed; a bad vram_gb was dropped without a word."""
    (tmp_path / "local_models.json").write_text(json.dumps({"default": "a:1", "models": [
        {"model": "a:1", "label": "A", "size_gb": "6.6 GB", "vram_gb": "lots", "rank": True, "note": ""},
        {"model": "b:1", "label": "B", "size_gb": 2, "vram_gb": 2.5, "rank": 1, "note": ""}]}), encoding="utf-8")
    with caplog.at_level("WARNING"):
        models = hp.translation_models(tmp_path)
    a, b = models
    assert (a["size_gb"], a["vram_gb"], a["rank"]) == (0.0, None, None)
    assert (b["size_gb"], b["vram_gb"], b["rank"]) == (2.0, 2.5, 1)
    assert sum("a:1" in r.getMessage() for r in caplog.records) == 3


def test_funasr_models_match_the_app():
    from model_manager import FUNASR_MODEL_PROFILES

    assert {k.split(":", 1)[1] for k in hp.ASR_VRAM_GB if k.startswith("funasr:")} == set(FUNASR_MODEL_PROFILES)


@pytest.mark.parametrize("settings, key, vram", [
    ({"asr_engine": "whisper", "whisper_model_size": TURBO, "asr_device": "cuda:0"}, "whisper:turbo", 2.5),
    ({"asr_engine": "whisper", "whisper_model_size": "medium"}, "whisper:medium", 2.0),
    ({"asr_engine": "funasr", "funasr_model": "sensevoice-small"}, "funasr:sensevoice-small", 0.7),
    ({"asr_engine": "sensevoice"}, "funasr:sensevoice-small", 0.7),            # legacy engine name
    ({"asr_engine": "funasr", "funasr_model": "funasr-mlt-nano-2512"}, "funasr:funasr-mlt-nano-2512", 3.7),
    ({"asr_engine": "anime-whisper"}, "anime-whisper", 1.8),
    ({"asr_engine": "remote-whisper"}, "remote-whisper", 0.0),                 # runs on another computer
    ({"asr_engine": "whisper", "whisper_model_size": TURBO, "asr_device": "cpu"}, "whisper:turbo", 0.0),
    ({"asr_engine": "whisper", "whisper_model_size": r"D:\models\my-own-whisper"}, "whisper:large-v3", 3.6),
    ({}, "funasr:sensevoice-small", 0.7),                                       # config.yaml default
])
def test_speech_model_is_read_from_the_settings(settings, key, vram):
    asr = hp.current_asr(settings)
    assert asr["key"] == key and asr["vram_gb"] == vram
    assert asr["label"]


# ---------------------------------------------------------------- judging one pair

@pytest.mark.parametrize("total, asr, tl, level", [
    (11.9, 2.5, 6.3, "ok"),        # RTX 5070 12 GB: Whisper turbo + Qwen3.5 9B (measured 10.2 GB)
    (11.9, 3.7, 6.3, "tight"),     # MLT-Nano + 9B: measured 11.4 GB, the benchmark marks it
    (8.0, 2.5, 6.3, "too_big"),    # 8 GB card cannot hold turbo + 9B
    (8.0, 2.5, 3.7, "tight"),
    (8.0, 0.7, 4.2, "ok"),
])
def test_level_of_a_pair(total, asr, tl, level):
    assert hp.level_for(total, asr + tl) == level


def test_unknown_card_or_model_is_not_judged():
    assert hp.level_for(None, 8.0) == "unknown"
    assert hp.level_for(12.0, None) == "unknown"


# ---------------------------------------------------------------- launcher window

def test_launcher_advice_marks_each_model_and_picks_the_best_that_fits():
    settings = {"asr_engine": "whisper", "whisper_model_size": TURBO, "asr_device": "cuda:0"}
    a = hp.launcher_advice(settings, hp.translation_models(ROOT), _gpu(11.9))
    levels = {m["model"]: m["level"] for m in a["models"]}
    assert levels == {"qwen3.5:9b": "ok", "qwen3.5:4b": "ok", SAKURA: "ok"}
    assert a["recommended"] == "qwen3.5:9b" and a["asr"]["key"] == "whisper:turbo"

    a = hp.launcher_advice(settings, hp.translation_models(ROOT), _gpu(8.0))
    levels = {m["model"]: m["level"] for m in a["models"]}
    assert levels["qwen3.5:9b"] == "too_big" and levels[SAKURA] == "too_big" and levels["qwen3.5:4b"] == "tight"
    assert a["recommended"] == "qwen3.5:4b"  # nothing fits comfortably: the one that still fits
    need = {m["model"]: m["total_need_gb"] for m in a["models"]}
    assert need["qwen3.5:9b"] == pytest.approx(2.5 + 6.3 + hp.RESERVE_GB)


def test_launcher_advice_follows_the_speech_model_in_use():
    """With the light SenseVoice the same 8 GB card can run Sakura comfortably."""
    a = hp.launcher_advice({"asr_engine": "funasr", "funasr_model": "sensevoice-small"},
                           hp.translation_models(ROOT), _gpu(8.0))
    assert a["recommended"] == SAKURA


def test_launcher_advice_without_a_readable_card_marks_nothing():
    a = hp.launcher_advice({}, hp.translation_models(ROOT), None)
    assert a["gpu"] is None and a["recommended"] is None
    assert {m["level"] for m in a["models"]} == {"unknown"}


def test_launcher_advice_when_nothing_fits():
    a = hp.launcher_advice({"asr_engine": "whisper", "whisper_model_size": TURBO}, hp.translation_models(ROOT), _gpu(4.0))
    assert a["recommended"] is None and {m["level"] for m in a["models"]} == {"too_big"}


# ---------------------------------------------------------------- first install

@pytest.mark.parametrize("total, asr, model", [
    (11.9, "whisper-turbo", "qwen3.5:9b"),   # 12 GB: the full setup
    (10.0, "whisper-turbo", SAKURA),          # 10 GB: 9B does not fit next to turbo
    (8.0, "sensevoice", SAKURA),              # 8 GB: turbo only fits "tight": take the safe pair
    (6.0, "sensevoice", "qwen3.5:4b"),        # 6 GB: the smallest pair, close to the limit
])
def test_install_picks_the_best_pair_the_card_can_run(total, asr, model):
    c = hp.install_choice(total, hp.translation_models(ROOT))
    assert (c["asr_choice"], c["model"]) == (asr, model)
    assert c["level"] in ("ok", "tight")


def test_install_on_a_card_too_small_for_any_pair_says_so():
    c = hp.install_choice(4.0, hp.translation_models(ROOT))
    assert c["level"] == "too_big" and (c["asr_choice"], c["model"]) == ("sensevoice", "qwen3.5:4b")
    assert "雲端" in c["message"]


def test_install_without_a_known_card_keeps_the_usual_defaults():
    c = hp.install_choice(None, hp.translation_models(ROOT), hp.default_model(ROOT))
    assert (c["asr_choice"], c["model"], c["level"]) == ("whisper-turbo", "qwen3.5:9b", "unknown")
    # the catalog's "default", not whichever model happens to be listed first
    c = hp.install_choice(None, list(reversed(hp.translation_models(ROOT))), "qwen3.5:9b")
    assert c["model"] == "qwen3.5:9b"


@pytest.mark.parametrize("device, index", [
    ("cuda:1 (NVIDIA GeForce RTX 3060)", 1), ("cuda:0", 0), ("cuda", 0), ("cpu", 0),
    ("cuda:", 0), ("cuda:x", 0),  # unreadable: GPU 0 instead of an exception
])
def test_gpu_number_comes_from_the_asr_device(device, index):
    assert hp.gpu_index({"asr_device": device}) == index


def test_install_settings_follow_the_chosen_speech_model():
    from factory_defaults import install_settings

    s = install_settings("medium", "http://127.0.0.1:11434/v1", SAKURA, asr="sensevoice")
    assert s["asr_engine"] == "funasr" and s["funasr_model"] == "sensevoice-small"
    assert s["models"][0]["model"] == SAKURA
    s = install_settings(TURBO, "http://127.0.0.1:11434/v1", "qwen3.5:9b")
    assert s["asr_engine"] == "whisper" and s["whisper_model_size"] == TURBO
    with pytest.raises(ValueError):
        install_settings(TURBO, "http://127.0.0.1:11434/v1", "qwen3.5:9b", asr="whisper-v9")


def test_installed_model_entry_carries_the_catalog_settings():
    """Review finding: the install wrote a fixed entry (thinking_style "openai", no
    prompt_style), so a Sakura chosen for an 8 GB card started with the wrong settings."""
    from factory_defaults import install_settings

    entry = install_settings("medium", "http://127.0.0.1:11434/v1", SAKURA, asr="sensevoice")["models"][0]
    assert entry["thinking_style"] == "off" and entry["prompt_style"] == "sakura"
    assert entry["api_base"] == "http://127.0.0.1:11434/v1" and entry["model"] == SAKURA
    custom = install_settings("medium", "http://127.0.0.1:11434/v1", "my-own:7b")["models"][0]
    assert custom["model"] == "my-own:7b" and custom["thinking_style"] == "openai"


def test_install_tables_agree():
    from factory_defaults import INSTALL_ASR_ENGINES

    assert {choice for choice, _key in hp.INSTALL_ASR} == set(INSTALL_ASR_ENGINES)


# ---------------------------------------------------------------- free memory at start

def test_start_check_needs_the_pair_in_use():
    settings = {"asr_engine": "whisper", "whisper_model_size": TURBO}
    c = hp.startup_check(settings, "qwen3.5:4b", hp.translation_models(ROOT))
    assert c["free_need_gb"] == pytest.approx(2.5 + 3.7 + hp.FREE_MARGIN_GB)
    assert "Whisper" in c["asr_label"] and c["model_label"] == "Qwen3.5 4B"
    # a model outside the catalog: the speech model alone plus the largest catalog model
    c = hp.startup_check(settings, "my-own:model", hp.translation_models(ROOT))
    assert c["free_need_gb"] == pytest.approx(2.5 + 6.3 + hp.FREE_MARGIN_GB)


# ---------------------------------------------------------------- reading the card

def test_gpu_is_read_through_nvml(monkeypatch):
    import system_stats

    class Fake:
        def __init__(self, index):
            self.index = index

        def read(self):
            return system_stats.GpuStats(used_mb=2048, total_mb=12227, util_percent=3)

        def name(self):
            return "NVIDIA GeForce RTX 5070"

    monkeypatch.setattr(system_stats, "NvmlGpu", Fake)
    g = hp.detect_gpu(0)
    assert g == {"name": "NVIDIA GeForce RTX 5070", "total_gb": pytest.approx(11.94, abs=0.01),
                 "free_gb": pytest.approx(9.94, abs=0.01)}


def test_no_nvidia_driver_means_no_gpu(monkeypatch):
    import system_stats

    monkeypatch.setattr(system_stats, "_load_nvml", lambda: None)
    assert hp.detect_gpu(0) is None


# ---------------------------------------------------------------- command line used by the launcher

def _cli(*args):
    return subprocess.run([sys.executable, "-I", str(ROOT / "launcher" / "hardware_advice.py"), *args],
                          capture_output=True, text=True, encoding="utf-8", timeout=60)


def test_cli_launcher_and_install_answers(tmp_path):
    settings = tmp_path / "user_settings.json"
    settings.write_text(json.dumps({"asr_engine": "whisper", "whisper_model_size": TURBO}), encoding="utf-8")
    r = _cli("launcher", str(settings), "--total-mb", "8192")
    assert r.returncode == 0, r.stderr
    a = json.loads(r.stdout)
    assert a["gpu"]["total_gb"] == 8.0 and a["recommended"] == "qwen3.5:4b"
    r = _cli("install", "--total-mb", "10240")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["model"] == SAKURA
    r = _cli("startup", str(settings), "qwen3.5:9b", "--total-mb", "12227")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["free_need_mb"] == round((2.5 + 6.3 + hp.FREE_MARGIN_GB) * 1024)


def test_cli_with_a_missing_or_broken_settings_file_uses_the_defaults(tmp_path):
    r = _cli("launcher", str(tmp_path / "missing.json"), "--total-mb", "8192")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["asr"]["key"] == "funasr:sensevoice-small"
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    r = _cli("launcher", str(broken), "--total-mb", "8192")
    assert r.returncode == 0, r.stderr
    assert "user_settings" in r.stderr  # the reason is logged, the window still gets advice


def test_cli_rejects_bad_arguments():
    assert _cli("launcher").returncode == 2
    assert _cli("install", "--total-mb", "lots").returncode == 2
    assert _cli("teleport").returncode == 2
