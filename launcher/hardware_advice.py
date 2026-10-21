"""Hardware advice for the PowerShell launcher (logic: hardware_profile.py). Prints one
line of JSON (ASCII-only, so the console code page cannot garble it).

Usage:
  python -I hardware_advice.py launcher <user_settings.json> [--total-mb N]
      each catalog model's level next to the speech model in use, and the recommendation
  python -I hardware_advice.py install --total-mb N
      speech + translation defaults for a fresh install on a card with N MB
  python -I hardware_advice.py startup <user_settings.json> <model> [--total-mb N]
      free GPU memory the pair in use needs at start (free_need_mb, gpu_index)

--total-mb replaces the detected card (the installer passes nvidia-smi's number). A
missing or unreadable settings file means the app's defaults; the reason goes to stderr.
Exit code 2 = bad arguments, 1 = unexpected error."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hardware_profile as hp  # noqa: E402


def _settings(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[hardware_advice] cannot read user_settings {path}: {e}; using the defaults", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"[hardware_advice] user_settings {path} is not an object; using the defaults", file=sys.stderr)
        return {}
    return data


def _gpu(settings: dict, total_mb):
    if total_mb is not None:
        if total_mb <= 0:
            return None
        return {"name": "", "total_gb": total_mb / 1024, "free_gb": None}
    return hp.detect_gpu(hp.gpu_index(settings))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hardware_advice.py")
    parser.add_argument("command", choices=("launcher", "install", "startup"))
    parser.add_argument("args", nargs="*")
    parser.add_argument("--total-mb", type=int, default=None)
    opts = parser.parse_args(argv)
    need = {"launcher": 1, "install": 0, "startup": 2}[opts.command]
    if len(opts.args) != need:
        parser.error(f"{opts.command} takes {need} argument(s)")
    catalog = hp.translation_models()
    if opts.command == "install":
        total = opts.total_mb / 1024 if opts.total_mb and opts.total_mb > 0 else None
        result = hp.install_choice(total, catalog, hp.default_model())
    elif opts.command == "launcher":
        settings = _settings(opts.args[0])
        result = hp.launcher_advice(settings, catalog, _gpu(settings, opts.total_mb))
    else:
        settings = _settings(opts.args[0])
        result = hp.startup_check(settings, opts.args[1], catalog)
        result["free_need_mb"] = round(result["free_need_gb"] * 1024)
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:  # reported to the launcher, which then shows no advice
        print(f"[hardware_advice] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
