"""Local (Ollama) translation models: the catalog in local_models.json, keeping the
model entries in user_settings.json in step with what is installed, and remembering the
model chosen in the app for the next launch (launcher/selected_model.txt)."""
import copy
import json
import logging
import os
import re
from pathlib import Path

log = logging.getLogger("LiveTranslate.Models")

CATALOG_FILE = Path(__file__).with_name("local_models.json")
# Read by launcher/model.ps1; tests point this at a sandbox (tests/conftest.py)
SELECTED_MODEL_FILE = Path(__file__).parent / "launcher" / "selected_model.txt"

# Same rule as launcher/model.ps1 Test-LauncherModelName: the name reaches `ollama pull`
# and JSON request bodies, so nothing but a plain model tag is accepted.
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_AUTO_NAME_RE = re.compile(r"^Ollama [A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def _number(entry: dict, key: str, kind):
    """entry[key] as `kind` when it is a plain number, else None (bool is not a number
    here); a value that is present but not a number is logged."""
    value = entry.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        log.warning(f"Catalog model {entry.get('model')!r}: {key} {value!r} is not a number; ignored")
        return None
    return kind(value)


def load_catalog(path=None) -> dict:
    """{"default": tag, "models": [{"model", "label", "size_gb", "vram_gb", "rank", "note",
    "settings"}]}. vram_gb / rank are None when missing (the model is then not judged
    against the graphics card)."""
    data = json.loads(Path(path or CATALOG_FILE).read_text(encoding="utf-8"))
    models = []
    for c in data.get("models", []):
        if not (isinstance(c, dict) and isinstance(c.get("model"), str) and MODEL_NAME_RE.match(c["model"])):
            log.warning(f"Ignoring invalid catalog entry: {c!r}")
            continue
        models.append({
            "model": c["model"],
            "label": str(c.get("label") or c["model"]),
            "size_gb": _number(c, "size_gb", float) or 0.0,
            "vram_gb": _number(c, "vram_gb", float),
            "rank": _number(c, "rank", int),
            "note": str(c.get("note") or ""),
            "settings": dict(c.get("settings") or {}),
        })
    default = data.get("default")
    if not any(m["model"] == default for m in models):
        default = models[0]["model"] if models else None
    return {"default": default, "models": models}


def is_local_ollama(entry) -> bool:
    if not isinstance(entry, dict):
        return False
    return entry.get("api_key") == "ollama" or ":1143" in str(entry.get("api_base", ""))


def _new_entry(c: dict, api_base: str) -> dict:
    entry = {"name": c["label"], "api_base": api_base, "api_key": "ollama", "model": c["model"], "proxy": "none"}
    entry.update(copy.deepcopy(c["settings"]))
    return entry


def catalog_entry(model: str, api_base: str):
    """A user_settings model entry for catalog model `model` (its catalog settings
    included), or None when `model` is not in the catalog."""
    c = next((m for m in load_catalog()["models"] if m["model"] == model), None)
    return None if c is None else _new_entry(c, api_base)


def sync_local_models(settings, api_base: str, selected: str, installed) -> dict:
    """Return settings whose local model entries match the catalog models that are
    installed (plus the selected one, which the launcher is about to pull), all pointed at
    `api_base`, with the selected model active. Other entries (cloud APIs, custom local
    models) are kept; per-model tweaks of existing entries are kept.

    installed=None means "could not ask Ollama": then nothing is removed, so a transient
    failure never deletes entries or their tweaks."""
    if not (isinstance(selected, str) and MODEL_NAME_RE.match(selected)):
        raise ValueError(f"invalid model name: {selected!r}")
    s = copy.deepcopy(settings) if isinstance(settings, dict) else {}
    entries = [m for m in s.get("models", []) if isinstance(m, dict)] if isinstance(s.get("models"), list) else []
    known = installed is not None
    wanted = set(installed or []) | {selected}
    catalog = load_catalog()["models"]
    by_model = {c["model"]: c for c in catalog}

    result = []
    seen = set()
    for m in entries:
        if not is_local_ollama(m):
            result.append(m)
            continue
        m["api_base"] = api_base
        model = m.get("model")
        if model in by_model:
            if model in seen or (known and model not in wanted):
                continue  # duplicate, or uninstalled catalog model: switching to it would fail
            c = by_model[model]
            name = str(m.get("name", ""))
            if not name or _AUTO_NAME_RE.match(name) or name == c["label"]:
                m["name"] = c["label"]
            for k, v in c["settings"].items():  # e.g. prompt_style for Sakura
                m.setdefault(k, copy.deepcopy(v))
            seen.add(model)
        result.append(m)

    for c in catalog:
        if c["model"] in wanted and c["model"] not in seen:
            result.append(_new_entry(c, api_base))
            seen.add(c["model"])
    if selected not in seen and not any(is_local_ollama(m) and m.get("model") == selected for m in result):
        result.append({"name": f"Ollama {selected}", "api_base": api_base, "api_key": "ollama",
                       "model": selected, "proxy": "none"})

    s["models"] = result
    s["active_model"] = next(i for i, m in enumerate(result) if is_local_ollama(m) and m.get("model") == selected)
    return s


def remember_selected(model_config) -> bool:
    """Write the local model chosen in the app to launcher/selected_model.txt so the next
    launch pulls / warms the same one. Cloud models and invalid names are ignored."""
    if not is_local_ollama(model_config):
        return False
    model = model_config.get("model")
    if not (isinstance(model, str) and MODEL_NAME_RE.match(model)):
        log.warning(f"Not remembering invalid local model name: {model!r}")
        return False
    path = Path(SELECTED_MODEL_FILE)
    try:
        if path.exists() and path.read_text(encoding="utf-8-sig").strip() == model:
            return True
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(model.encode("utf-8"))
        os.replace(tmp, path)
        log.info(f"Selected translation model for the next launch: {model}")
        return True
    except OSError as e:
        log.warning(f"Could not save the selected model to {path}: {e}")
        return False


def ollama_root(api_base: str) -> str:
    """'http://127.0.0.1:11434/v1' -> 'http://127.0.0.1:11434'."""
    base = str(api_base).rstrip("/")
    return base[:-3] if base.endswith("/v1") else base
