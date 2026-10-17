"""Point the Ollama model entry in user_settings.json at the given API base.

Usage: python -I set_api_base.py <user_settings.json> <api_base> <model_name>
Only touches model entries whose api_key is "ollama" or whose api_base
already points at a local Ollama port. Everything else is left as-is.
"""
import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    path, api_base, model = sys.argv[1:4]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False
    for m in data.get("models", []):
        is_ollama = m.get("api_key") == "ollama" or ":1143" in str(m.get("api_base", ""))
        if not is_ollama:
            continue
        if m.get("api_base") != api_base:
            m["api_base"] = api_base
            changed = True
        if m.get("model") != model:
            m["model"] = model
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
