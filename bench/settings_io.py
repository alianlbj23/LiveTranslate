"""User settings: read-only access for the benchmark, the running-app check, and
"apply recommendation" (backup first, change only the needed keys, restore).

The app writes user_settings.json itself when it closes, so nothing is written while it
runs. The launcher rewrites active_model from launcher/selected_model.txt on every
start, so a local Ollama choice is written there too (local_models.remember_selected)."""
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from bench import paths

log = logging.getLogger("LiveTranslate.Bench.Settings")

ASR_SETTINGS = {
    "sensevoice": {"asr_engine": "funasr", "funasr_model": "sensevoice-small"},
    "mlt-nano": {"asr_engine": "funasr", "funasr_model": "funasr-mlt-nano-2512"},
    "anime-whisper": {"asr_engine": "anime-whisper"},
    "whisper-turbo": {"asr_engine": "whisper"},  # whisper_model_size = local turbo path
}


class SettingsError(RuntimeError):
    pass


def load_settings(path: Path | None = None) -> dict:
    path = Path(path or paths.SETTINGS_FILE)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise SettingsError(f"找不到設定檔 {path}，請先啟動一次 LiveTranslate") from e
    except ValueError as e:
        raise SettingsError(f"設定檔 {path} 格式錯誤：{e}") from e
    if not isinstance(data, dict):
        raise SettingsError(f"設定檔 {path} 內容不是物件")
    return data


def load_config(path: Path | None = None) -> dict:
    import yaml

    path = Path(path or paths.CONFIG_FILE)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SettingsError(f"{path} 內容不是物件")
    return data


def translation_models(settings: dict) -> list:
    models = settings.get("models")
    if not isinstance(models, list):
        return []
    return [m for m in models if isinstance(m, dict) and m.get("api_base") and m.get("model")]


def find_running_app() -> list:
    """PIDs of a LiveTranslate main.py started from this project folder."""
    import psutil

    root = str(paths.ROOT).lower()
    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = p.info.get("cmdline") or []
            if not name.startswith("python") or not any(str(a).endswith("main.py") for a in cmd):
                continue
            where = " ".join([str(p.info.get("cwd") or "")] + [str(a) for a in cmd]).lower()
            if root in where:
                found.append(p.info["pid"])
        except psutil.Error as e:
            log.debug(f"Skipping process {p.pid}: {e}")
    return found


def planned_changes(settings: dict, asr_id: str, vad_threshold: float, tl_entry: dict) -> dict:
    """{key: (old, new)} for the keys that would change."""
    from bench.asr_runner import find_turbo_path

    new = dict(ASR_SETTINGS[asr_id])
    if asr_id == "whisper-turbo":
        turbo = find_turbo_path()
        if not turbo:
            raise SettingsError("找不到 Whisper large-v3-turbo 模型資料夾")
        new["whisper_model_size"] = turbo
    new["vad_threshold"] = round(float(vad_threshold), 2)
    index = next((i for i, m in enumerate(settings.get("models", []))
                  if isinstance(m, dict) and m.get("name") == tl_entry.get("name")
                  and m.get("model") == tl_entry.get("model")), None)
    if index is None:
        raise SettingsError(f"設定檔裡找不到翻譯模型「{tl_entry.get('name')}」")
    new["active_model"] = index
    return {k: (settings.get(k), v) for k, v in new.items() if settings.get(k) != v}


def _atomic_write_json(path: Path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def backups_dir() -> Path:
    return paths.results_dir() / "settings_backups"


def backup(settings_path: Path, selected_file: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = backups_dir() / stamp
    target.mkdir(parents=True)
    shutil.copy2(settings_path, target / "user_settings.json")
    if json.loads((target / "user_settings.json").read_text(encoding="utf-8")) != \
            json.loads(Path(settings_path).read_text(encoding="utf-8")):
        raise SettingsError("備份驗證失敗，沒有修改設定")
    if Path(selected_file).exists():
        shutil.copy2(selected_file, target / "selected_model.txt")
    return target


def apply_changes(changes: dict, tl_entry: dict, settings_path: Path | None = None,
                  selected_file: Path | None = None) -> Path:
    """Back up, then write only the changed keys. Refuses while LiveTranslate runs."""
    import local_models

    if find_running_app():
        raise SettingsError("LiveTranslate 正在執行。請先從系統匣按「結束」，它關閉時會存設定、蓋掉這次套用。")
    settings_path = Path(settings_path or paths.SETTINGS_FILE)
    selected_file = Path(selected_file or local_models.SELECTED_MODEL_FILE)
    settings = load_settings(settings_path)
    saved = backup(settings_path, selected_file)
    for key, (_old, new) in changes.items():
        settings[key] = new
    _atomic_write_json(settings_path, settings)
    if load_settings(settings_path) != settings:
        raise SettingsError(f"寫入後讀回不一致；原本的設定在 {saved}")
    if local_models.is_local_ollama(tl_entry):
        local_models.remember_selected(tl_entry)
    log.info(f"Applied {sorted(changes)}; backup in {saved}")
    return saved


def list_backups() -> list:
    root = backups_dir()
    if not root.is_dir():
        return []
    return sorted((d for d in root.iterdir() if (d / "user_settings.json").exists()), reverse=True)


def restore(backup_dir: Path, settings_path: Path | None = None, selected_file: Path | None = None):
    import local_models

    if find_running_app():
        raise SettingsError("LiveTranslate 正在執行，請先關閉再還原。")
    settings_path = Path(settings_path or paths.SETTINGS_FILE)
    selected_file = Path(selected_file or local_models.SELECTED_MODEL_FILE)
    data = json.loads((backup_dir / "user_settings.json").read_text(encoding="utf-8"))
    _atomic_write_json(settings_path, data)
    sel = backup_dir / "selected_model.txt"
    if sel.exists():
        tmp = selected_file.with_name(selected_file.name + ".tmp")
        tmp.write_bytes(sel.read_bytes())
        os.replace(tmp, selected_file)


# ---------------------------------------------------------------- control panel steps

def ui_labels() -> dict:
    import yaml

    defaults = {"tab_vad_asr": "VAD / ASR", "tab_translation": "翻譯", "label_engine": "引擎:",
                "asr_fast": "快速", "asr_accurate": "準確", "label_funasr_model": "FunASR 模型:",
                "group_silero_threshold": "Silero VAD 閾值", "label_threshold": "閾值:",
                "group_active_model": "當前模型 (用於翻譯管線)", "btn_apply": "應用", "label_model": "模型:"}
    try:
        data = yaml.safe_load((paths.ROOT / "i18n" / "zh_TW.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        log.warning(f"Could not read UI labels: {e}")
        data = {}
    return {k: str(data.get(k, v)) for k, v in defaults.items()}


def setup_steps(asr_id: str, vad_threshold: float, tl_name: str, rec_id: str) -> list:
    from model_manager import funasr_display_name

    L = ui_labels()
    if asr_id in ("sensevoice", "mlt-nano"):
        engine = (f"「{L['tab_vad_asr']}」頁 →「{L['label_engine']}」選「[{L['asr_fast']}] FunASR」→"
                  f"「{L['label_funasr_model']}」選「{funasr_display_name(ASR_SETTINGS[asr_id]['funasr_model'])}」")
    elif asr_id == "whisper-turbo":
        engine = (f"「{L['tab_vad_asr']}」頁 →「{L['label_engine']}」選「[{L['asr_accurate']}] Whisper (faster-whisper)」→"
                  f"「{L['label_model']}」選「mobiuslabsgmbh/faster-whisper-large-v3-turbo」")
    else:
        engine = f"「{L['tab_vad_asr']}」頁 →「{L['label_engine']}」選「Anime-Whisper (ja, anime/galgame)」"
    return [
        engine,
        f"「{L['group_silero_threshold']}」→「{L['label_threshold']}」設成 {vad_threshold:.2f}",
        f"「{L['tab_translation']}」頁 →「{L['group_active_model']}」選「{tl_name}」→ 按「{L['btn_apply']}」"
        f"（也可以在對話記錄視窗上方的模型選單直接切換）",
        f"或在啟動視窗按「最佳組合評測」→ 選 5，套用 {rec_id}（會先備份目前的設定）",
    ]
