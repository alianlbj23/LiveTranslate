"""Run folder, resumable state and result logs.

bench_results/<YYYYMMDD_HHMMSS>/
  plan.json      what this run measures (never changes; no API keys)
  state.json     task list and frozen stage results (atomic writes)
  results/<kind>.jsonl   one record per line, flushed + fsynced; a cut-off last line
                         (crash mid-write) is ignored when reading
  report.html, results.csv, recommendations.json
bench_results/.running.lock holds the PID of the run in progress."""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from bench import paths

log = logging.getLogger("LiveTranslate.Bench.State")
SCHEMA = 1


def atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class RunStore:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.plan = json.loads((self.dir / "plan.json").read_text(encoding="utf-8"))
        state_file = self.dir / "state.json"
        if state_file.exists():
            self.state = json.loads(state_file.read_text(encoding="utf-8"))
        else:
            self.state = {"schema": SCHEMA, "status": "running", "tasks": [], "frozen": {}, "timing": {}}

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
    def latest_unfinished(cls):
        for d in cls.list_runs():
            try:
                store = cls(d)
            except (OSError, ValueError) as e:
                log.warning(f"Skipping unreadable run {d}: {e}")
                continue
            if store.state.get("status") != "done":
                return store
        return None

    # -------------------------------------------------------------- state
    def save(self):
        atomic_write_json(self.dir / "state.json", self.state)

    def task(self, task_id: str, **defaults) -> dict:
        for t in self.state["tasks"]:
            if t["id"] == task_id:
                return t
        t = {"id": task_id, "status": "pending", "items_done": 0, "items_total": 0,
             "attempts": 0, "reason": None, "elapsed_s": 0.0}
        t.update(defaults)
        self.state["tasks"].append(t)
        self.save()
        return t

    def set_task(self, task_id: str, **fields):
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
        # a line cut off by a crash must not swallow the next record
        broken = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
        with open(path, "a", encoding="utf-8") as f:
            f.write(("\n" if broken else "") + json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read(self, kind: str) -> list:
        path = self.dir / "results" / f"{kind}.jsonl"
        if not path.exists():
            return []
        out = []
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                if i == len(lines) - 1:
                    log.warning(f"Ignoring a cut-off last line in {path.name} (interrupted write)")
                else:
                    log.warning(f"Ignoring a corrupt line {i + 1} in {path.name}")
        return out

    def done_keys(self, kind: str, task_id: str) -> set:
        return {r.get("key") for r in self.read(kind) if r.get("task") == task_id}


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
