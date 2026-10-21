"""Write the initial user_settings.json for the Japanese -> Taiwanese Mandarin setup.

Usage: python -I write_settings.py <project_root> <whisper_model_path_or_size> <api_base> <model_name> [asr]

asr: whisper-turbo (default) or sensevoice, as chosen for the graphics card
(launcher/hardware_advice.py install). Existing user_settings.json is merged (our keys
win) so overlay positions etc. survive. The values come from factory_defaults.py in the
project root.
"""
import json
import os
import sys


def main() -> int:
    if len(sys.argv) not in (5, 6):
        print(__doc__)
        return 2
    root, whisper_model, api_base, model_name = sys.argv[1:5]
    asr = sys.argv[5] if len(sys.argv) == 6 else "whisper-turbo"
    path = os.path.join(root, "user_settings.json")

    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # The values live in factory_defaults.py in this project's root (shared with the
    # control panel's factory reset); `root` is only where user_settings.json goes.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from factory_defaults import install_settings

    try:
        data.update(install_settings(whisper_model, api_base, model_name, asr=asr))
    except ValueError as e:
        print(f"[write_settings] {e}", file=sys.stderr)
        return 2

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    print(f"[write_settings] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
