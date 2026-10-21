"""Run folder, resumable state and result logs.

bench_results/<YYYYMMDD_HHMMSS>/
  plan.json      what this run measures (never changes; no API keys)
  state.json     task list and frozen stage results (atomic writes)
  results/<kind>.jsonl   one record per line, flushed + fsynced; a cut-off last line
                         (crash mid-write) is ignored when reading and removed before
                         the next record is appended
  bench.log      everything the run logged (the console points here on errors)
  ollama_serve.log, gpu_watchdog.log   only when the benchmark started Ollama itself
  report.html, results.csv, recommendations.json
bench_results/.running.lock holds the PID of the run in progress.

Resume rules: a task marked done or failed is not run again, except that failed tasks
are reset when the user agrees (cli.cmd_resume) and judge tasks always are. Inside a
task that runs again only the items that are not final (is_final) are redone."""
import copy
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from bench import paths

log = logging.getLogger("LiveTranslate.Bench.State")
SCHEMA = 1
RUN_STATUSES = ("running", "done", "interrupted", "error")
TASK_STATUSES = ("pending", "running", "done", "failed")
FINISHED = ("done", "failed")
TASK_FIELDS = ("status", "items_done", "items_total", "attempts", "reason")
# Values a plan.json may lack (hand-written plans, plans of older versions); readers
# use plan[key] so a typo fails loudly instead of reading None
PLAN_DEFAULTS = {
    "schema": SCHEMA, "quick": False, "stages": [1, 2, 3], "datasets": [], "clip_counts": {}, "data_ids": {},
    "asr": [], "download_missing": False, "reuse_stage1_from": None, "recordings": [], "tl_models": [],
    "judges": [], "judge_mode": "single", "together_clips": 20, "restart_ollama": False, "tl_timeout_s": 10.0,
    "settings_snapshot": {},
}


def atomic_write_json(path: Path, data, indent: int = 1):
    """Write to <name>.tmp, fsync, then replace: a crash leaves the old file or the new
    one, never half of it. Raises OSError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class RunStore:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        loaded = json.loads((self.dir / "plan.json").read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{self.dir / 'plan.json'} is not a plan")
        self.plan = {**copy.deepcopy(PLAN_DEFAULTS), **loaded}
        state_file = self.dir / "state.json"
        if state_file.exists():
            self.state = json.loads(state_file.read_text(encoding="utf-8"))
        else:
            self.state = {"schema": SCHEMA, "status": "running", "tasks": [], "frozen": {}, "timing": {}}
        for name, data in (("plan", self.plan), ("state", self.state)):
            if data.get("schema") != SCHEMA:
                log.warning(f"{self.dir.name}: {name} schema {data.get('schema')} (this version writes {SCHEMA})")

    # -------------------------------------------------------------- creation / lookup
    @classmethod
    def create(cls, plan: dict, now=None) -> "RunStore":
        stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
        run_dir = paths.results_dir() / stamp
        n = 1
        while run_dir.exists():
            n += 1
            run_dir = paths.results_dir() / f"{stamp}_{n}"
        (run_dir / "results").mkdir(parents=True)
        atomic_write_json(run_dir / "plan.json", plan)
        store = cls(run_dir)
        store.save()
        return store

    @staticmethod
    def list_runs() -> list:
        root = paths.results_dir()
        if not root.is_dir():
            return []
        runs = [d for d in root.iterdir() if d.is_dir() and (d / "plan.json").exists()]
        return sorted(runs, key=lambda d: d.name, reverse=True)

    @classmethod
    def readable_runs(cls):
        """RunStores newest first; unreadable run folders are logged and skipped."""
        for d in cls.list_runs():
            try:
                store = cls(d)
            except (OSError, ValueError) as e:
                log.warning(f"Skipping unreadable run {d}: {e}")
                continue
            yield store

    @classmethod
    def latest_unfinished(cls):
        return next((s for s in cls.readable_runs() if s.state.get("status") != "done"), None)

    # -------------------------------------------------------------- state
    def save(self):
        atomic_write_json(self.dir / "state.json", self.state)

    def set_status(self, status: str, error: str | None = None):
        if status not in RUN_STATUSES:
            raise ValueError(f"unknown run status {status!r}")
        self.state["status"] = status
        if error is None:
            self.state.pop("error", None)
        else:
            self.state["error"] = error[:500]
        self.save()

    def task(self, task_id: str) -> dict:
        """The task's entry; a task seen for the first time is added as pending (and saved)."""
        for t in self.state["tasks"]:
            if t["id"] == task_id:
                return t
        t = {"id": task_id, "status": "pending", "items_done": 0, "items_total": 0, "attempts": 0, "reason": None}
        self.state["tasks"].append(t)
        self.save()
        return t

    def finished(self, task_id: str) -> bool:
        return self.task(task_id)["status"] in FINISHED

    def set_task(self, task_id: str, **fields):
        unknown = set(fields) - set(TASK_FIELDS)
        if unknown:
            raise ValueError(f"unknown task fields {sorted(unknown)}")
        if "status" in fields and fields["status"] not in TASK_STATUSES:
            raise ValueError(f"unknown task status {fields['status']!r}")
        t = self.task(task_id)
        t.update(fields)
        self.save()
        return t

    def frozen(self, key, default=None):
        return self.state["frozen"].get(key, default)

    def freeze(self, key, value):
        self.state["frozen"][key] = value
        self.save()

    # -------------------------------------------------------------- results
    def append(self, kind: str, record: dict):
        path = self.dir / "results" / f"{kind}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            _drop_cut_off_tail(path)  # a crash mid-write must not swallow the next record
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read(self, kind: str) -> list:
        """Records in order. Lines are split on "\\n" only (a record may hold U+2028) and
        decoded one by one, so a line cut inside a character costs that line only."""
        path = self.dir / "results" / f"{kind}.jsonl"
        if not path.exists():
            return []
        out = []
        lines = path.read_bytes().split(b"\n")
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line.decode("utf-8")))
            except ValueError:  # UnicodeDecodeError is a ValueError too
                if i == len(lines) - 1:
                    log.warning(f"Ignoring a cut-off last line in {path.name} (interrupted write)")
                else:
                    log.warning(f"Ignoring a corrupt line {i + 1} in {path.name}")
        return out

    def final_keys(self, kind: str, task_id: str) -> set:
        """Items of a task whose latest record needs no retry (is_final)."""
        latest = {}
        for r in self.read(kind):
            if r.get("task") == task_id:
                latest[r.get("key")] = r
        return {k for k, r in latest.items() if is_final(kind, r)}


def _drop_cut_off_tail(path: Path):
    """Cut a log back to its last complete line (a crash can leave half a record)."""
    with open(path, "r+b") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        f.seek(0)
        data = f.read()
        keep = data.rfind(b"\n") + 1
        log.warning(f"Removing a cut-off last line from {path.name}: {data[keep:keep + 80]!r}")
        f.truncate(keep)


def is_final(kind: str, record: dict) -> bool:
    """A record that needs no retry when its task runs again. Failures under load
    (together) are measurements themselves: replaying them alone would hide them."""
    from bench.asr_runner import ASR_EMPTY

    if kind == "asr":
        return not record.get("error")
    if kind == "together":
        return True
    if kind == "tl":
        return bool(record.get("ok")) or record.get("error") == ASR_EMPTY
    if kind == "judge":
        return record.get("scores") is not None
    if kind == "vad":
        return record.get("judged_windows") == record.get("windows")
    return True


# ---------------------------------------------------------------- lock

def lock_path() -> Path:
    return paths.results_dir() / ".running.lock"


def running_pid():
    """PID of a benchmark that is running now, or None (a stale lock is ignored)."""
    import psutil

    try:
        pid = int(lock_path().read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    if pid == os.getpid() or not psutil.pid_exists(pid):
        return None
    try:
        cmd = " ".join(psutil.Process(pid).cmdline())
    except psutil.Error as e:  # cannot inspect it: assume it is a benchmark (safe side)
        log.debug(f"Cannot inspect lock owner {pid}: {e}")
        return pid
    # PIDs are reused: a lock whose PID now belongs to another program is stale
    return pid if "bench_run.py" in cmd or "pytest" in cmd else None


def acquire_lock() -> bool:
    if running_pid():
        return False
    lock_path().parent.mkdir(parents=True, exist_ok=True)
    lock_path().write_text(str(os.getpid()), encoding="ascii")
    return True


def release_lock():
    try:
        if lock_path().read_text(encoding="ascii").strip() == str(os.getpid()):
            lock_path().unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        log.warning(f"Could not remove the run lock: {e}")
