"""Environment check (Settings → Home → Maintenance; it used to be the launcher's console
"check environment"): graphics driver, PyTorch and CTranslate2 with CUDA, Ollama, and a
test translation with the local model in use. No Qt: env_check_ui runs the checks on a
thread and shows each result as it arrives.

The CUDA checks run in a child Python without a window, so the app itself never creates a
CUDA context (it would hold a few hundred MB of graphics memory)."""
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import local_models
import ollama_server
from i18n import t

log = logging.getLogger("LiveTranslate.EnvCheck")

ROOT = Path(__file__).resolve().parent
OK, WARN, FAIL = "ok", "warn", "fail"
TEST_SENTENCE = "今日はいい天気ですね"


@dataclass
class Check:
    title: str
    status: str   # OK / WARN / FAIL
    detail: str


def _python() -> str:
    """The venv's console-less interpreter when there is one (the app may run under pythonw)."""
    for name in ("python.exe", "pythonw.exe"):
        p = ROOT / ".venv" / "Scripts" / name
        if p.exists():
            return str(p)
    exe = Path(sys.executable)
    console = exe.with_name("python.exe")
    if exe.name.lower() == "pythonw.exe" and console.exists():
        return str(console)  # pythonw has no stdout: the check's JSON would be lost
    return sys.executable


def _run(args, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def check_driver(run=_run) -> Check:
    title = t("env_driver")
    try:
        r = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total",
                 "--format=csv,noheader,nounits"], 15)
    except FileNotFoundError:
        return Check(title, FAIL, t("env_driver_missing"))
    except subprocess.TimeoutExpired:
        return Check(title, FAIL, t("env_timeout").format(s=15))
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    if r.returncode != 0 or not lines:
        return Check(title, FAIL, (r.stderr or r.stdout).strip()[:300] or t("env_driver_missing"))
    rows = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) == 4:
            name, driver, used, total = parts
            try:
                rows.append(t("env_driver_row").format(name=name, driver=driver, used=int(used) / 1024,
                                                       total=int(total) / 1024))
                continue
            except ValueError:
                pass
        rows.append(ln.strip())
    return Check(title, OK, "\n".join(rows))


_TORCH = ("import json, torch; ok = torch.cuda.is_available(); "
          "print(json.dumps({'torch': torch.__version__, 'cuda': ok, "
          "'device': torch.cuda.get_device_name(0) if ok else ''}))")
_CT2 = ("import json, ctranslate2, faster_whisper; n = ctranslate2.get_cuda_device_count(); "
        "print(json.dumps({'ct2': ctranslate2.__version__, 'fw': faster_whisper.__version__, 'devices': n, "
        "'types': sorted(ctranslate2.get_supported_compute_types('cuda')) if n else []}))")


def _child_json(code: str, title: str, run, timeout: float = 120):
    """(dict, None) or (None, Check) for a failure."""
    try:
        r = run([_python(), "-I", "-c", code], timeout)
    except subprocess.TimeoutExpired:
        return None, Check(title, FAIL, t("env_timeout").format(s=int(timeout)))
    except OSError as e:
        return None, Check(title, FAIL, str(e))
    try:
        return json.loads(r.stdout.strip().splitlines()[-1]), None
    except (ValueError, IndexError):
        return None, Check(title, FAIL, (r.stderr or r.stdout).strip()[-400:] or f"exit code {r.returncode}")


def check_torch(run=_run) -> Check:
    title = t("env_torch")
    data, failed = _child_json(_TORCH, title, run)
    if failed:
        return failed
    if not data.get("cuda"):
        return Check(title, WARN, t("env_torch_no_cuda").format(version=data.get("torch", "?")))
    return Check(title, OK, f"torch {data['torch']}｜{data['device']}")


def check_ctranslate2(run=_run) -> Check:
    title = t("env_ct2")
    data, failed = _child_json(_CT2, title, run)
    if failed:
        return failed
    detail = f"ctranslate2 {data['ct2']}｜faster-whisper {data['fw']}"
    if not data.get("devices"):
        return Check(title, WARN, detail + "\n" + t("env_ct2_no_cuda"))
    return Check(title, OK, detail + "\n" + t("env_ct2_types").format(types=", ".join(data.get("types", []))))


def check_ollama() -> Check:
    import httpx

    title = "Ollama"
    rows, status = [], WARN
    for port in (ollama_server.DEFAULT_PORT, ollama_server.ALT_PORT):
        server = ollama_server.Ollama(ollama_server.root_url(port))
        version = server.version() if ollama_server.port_open(port) else None
        if version is None:
            rows.append(t("env_ollama_port_none").format(port=port))
            continue
        status = OK
        owner, _pid = ollama_server.port_owner(port)
        try:
            names = ", ".join(sorted(server.installed())) or t("env_none")
        except (httpx.HTTPError, ValueError) as e:  # shown in the result; not fatal
            names = f"?（{e}）"
        rows.append(t("env_ollama_port").format(port=port, version=version, owner=owner, models=names))
    if not ollama_server.OLLAMA_EXE.exists():
        rows.append(t("env_ollama_exe_missing").format(path=ollama_server.OLLAMA_EXE))
        if status != OK:
            status = FAIL
    elif status != OK:
        rows.append(t("env_ollama_starts_with_app"))
    return Check(title, status, "\n".join(rows))


def check_translation(model_entry) -> Check:
    """One test sentence through the local Ollama model in use (thinking off)."""
    import httpx

    title = t("env_translation")
    if not local_models.is_local_ollama(model_entry or {}):
        return Check(title, WARN, t("env_translation_cloud"))
    model = model_entry.get("model", "")
    root = local_models.ollama_root(str(model_entry.get("api_base", "")))
    try:
        port = int(root.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        port = ollama_server.DEFAULT_PORT
    if not ollama_server.port_open(port):
        return Check(title, WARN, t("env_translation_no_ollama"))
    body = {"model": model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": t("env_translation_prompt").format(text=TEST_SENTENCE)}]}
    t0 = time.perf_counter()
    try:
        with httpx.Client(trust_env=False, timeout=httpx.Timeout(180.0, connect=3.0)) as client:
            resp = client.post(f"{root}/api/chat", json=body)
        resp.raise_for_status()
        text = str(resp.json().get("message", {}).get("content", "")).strip()
    except (httpx.HTTPError, ValueError) as e:
        return Check(title, FAIL, t("env_translation_failed").format(model=model, e=e))
    ms = int((time.perf_counter() - t0) * 1000)
    if not text:
        return Check(title, WARN, t("env_translation_empty").format(model=model, ms=ms))
    return Check(title, OK, t("env_translation_ok").format(model=model, text=text, ms=ms))


def run_all(model_entry, on_result, run=_run):
    """Run every check in order and call on_result(Check) as each finishes. A check that
    crashes is reported as failed instead of stopping the others."""
    checks = [lambda: check_driver(run), lambda: check_torch(run), lambda: check_ctranslate2(run),
              check_ollama, lambda: check_translation(model_entry)]
    for check in checks:
        try:
            result = check()
        except Exception as e:  # reported to the user and logged, never silent
            log.exception("Environment check crashed")
            result = Check(t("env_crashed"), FAIL, f"{type(e).__name__}: {e}")
        on_result(result)


def as_text(results) -> str:
    """Plain text of the results (the dialog's "copy" button), for an issue or a chat."""
    mark = {OK: "✓", WARN: "!", FAIL: "✗"}
    return "\n\n".join(f"[{mark.get(r.status, '?')}] {r.title}\n{r.detail}" for r in results)
