"""Entry point of the best-combination benchmark (launcher/bench_run.py). The benchmark
window (bench/window.py, tests/test_bench_window.py) runs it; the console menu is gone."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _entry(*args, env_dir=None):
    import os

    env = dict(os.environ)
    if env_dir:
        env["LIVETRANSLATE_BENCH_DIR"] = str(env_dir)
    return subprocess.run([sys.executable, "-I", str(ROOT / "launcher" / "bench_run.py"), *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, env=env)


def test_entry_point_rejects_unknown_actions():
    r = _entry("nonsense")
    assert r.returncode == 2 and "status" in r.stdout


def test_entry_point_status_shows_the_data_state(tmp_path):
    r = _entry("status", env_dir=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "測試資料" in r.stdout and "動畫 0/200" in r.stdout and "自己的錄音：0 檔" in r.stdout


def test_bench_data_folders_are_git_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "bench_clips/" in ignore and "bench_results/" in ignore
