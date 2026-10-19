"""Point the Ollama model entry in user_settings.json at the given API base.

Usage: python -I set_api_base.py <user_settings.json> <api_base> <model_name>
Only touches model entries whose api_key is "ollama" or whose api_base
already points at a local Ollama port. Everything else is left as-is.
An auto-generated display name ("Ollama <model>") follows the model, so the
overlay's model combo shows what is actually used; custom names are kept.
"""
import json
import os
import re
import sys

# Same rule as launcher/model.ps1 Test-LauncherModelName
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_AUTO_NAME_RE = re.compile(r"^Ollama [A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    path, api_base, model = sys.argv[1:4]
    if not _MODEL_RE.match(model):
        print(f"[set_api_base] invalid model name: {model!r}", file=sys.stderr)
        return 2
    if not re.match(r"^https?://[^\s\"']+$", api_base):
        print(f"[set_api_base] invalid api_base: {api_base!r}", file=sys.stderr)
        return 2
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[set_api_base] cannot read {os.path.basename(path)}: {e}", file=sys.stderr)
        return 1
    if not isinstance(data, dict) or not isinstance(data.get("models", []), list):
        print(f"[set_api_base] unexpected structure in {os.path.basename(path)}", file=sys.stderr)
        return 1

    changed = False
    for m in data.get("models", []):
        if not isinstance(m, dict):
            continue
        is_ollama = m.get("api_key") == "ollama" or ":1143" in str(m.get("api_base", ""))
        if not is_ollama:
            continue
        if m.get("api_base") != api_base:
            m["api_base"] = api_base
            changed = True
        if m.get("model") != model:
            m["model"] = model
            changed = True
        name = str(m.get("name", ""))
        if _AUTO_NAME_RE.match(name) and name != f"Ollama {model}":
            m["name"] = f"Ollama {model}"
            changed = True

    if changed:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        print(f"[set_api_base] updated api_base -> {api_base}, model -> {model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
