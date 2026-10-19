"""Point the local Ollama model entries in user_settings.json at the launcher's port.

Usage: python -I set_api_base.py <user_settings.json> <api_base> <selected_model> [installed,models]

Keeps one entry per installed catalog model (local_models.json) so the model combo can
switch between them, adds the selected model if missing and makes it active. Cloud and
custom entries are kept. Without the installed list, Ollama's /api/tags is asked (3 s
timeout); if that fails, no entry is removed and the selected one is guaranteed.
"""
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import local_models  # noqa: E402


def _installed_from_ollama(api_base: str):
    """Installed model names, or None when Ollama could not be asked (nothing is removed then)."""
    url = local_models.ollama_root(api_base) + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", []) if isinstance(m, dict) and isinstance(m.get("name"), str)]
    except Exception as e:
        print(f"[set_api_base] could not list installed models ({e}); keeping all entries", file=sys.stderr)
        return None


def main() -> int:
    # The launcher reads this output as UTF-8 (Chinese model names, localized OS errors)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError) as e:
            print(f"[set_api_base] cannot switch output to UTF-8: {e}", file=sys.stderr)
    if len(sys.argv) not in (4, 5):
        print(__doc__)
        return 2
    path, api_base, model = sys.argv[1:4]
    if not local_models.MODEL_NAME_RE.match(model):
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
    if not isinstance(data, dict):
        print(f"[set_api_base] unexpected structure in {os.path.basename(path)}", file=sys.stderr)
        return 1

    if len(sys.argv) == 5:
        installed = [n for n in sys.argv[4].split(",") if local_models.MODEL_NAME_RE.match(n)]
    else:
        installed = _installed_from_ollama(api_base)
    new = local_models.sync_local_models(data, api_base, model, installed)
    if new != data:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(new, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        names = ", ".join(m.get("name", "?") for m in new["models"] if local_models.is_local_ollama(m))
        print(f"[set_api_base] api_base -> {api_base}, active -> {model}; local models: {names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
