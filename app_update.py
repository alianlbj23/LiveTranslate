"""Update check / update of a git checkout (Settings → Home → Maintenance, plus a quiet check
after start). Ported from launcher/update.ps1: fast-forward only, skipped while tracked
files have local changes. When requirements.txt changed, the packages are reinstalled by
launcher/update_window.ps1 after LiveTranslate has closed (pip cannot replace the DLLs a
running app holds); that window then starts LiveTranslate again. No Qt."""
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("LiveTranslate.Update")

ROOT = Path(__file__).resolve().parent
UPDATE_WINDOW = ROOT / "launcher" / "update_window.ps1"

# states
NOT_GIT = "not_git"            # release zip: no .git folder
NO_GIT_EXE = "no_git_exe"      # git is not installed
DETACHED = "detached"          # not on a branch
LOCAL_BRANCH = "local_branch"  # the branch exists only on this computer
OFFLINE = "offline"            # GitHub could not be reached
UP_TO_DATE = "up_to_date"
BEHIND = "behind"              # newer commits on GitHub, can update
DIRTY = "dirty"                # newer commits, but tracked files were edited here


@dataclass
class UpdateStatus:
    state: str
    branch: str = ""
    behind: int = 0
    detail: str = ""

    @property
    def can_update(self) -> bool:
        return self.state == BEHIND


def _git(args, root: Path, timeout: float) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")  # never wait for a password prompt
    return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=env,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _out(r: subprocess.CompletedProcess) -> str:
    return (r.stderr or r.stdout or "").strip()[:300]


def check(root: Path = ROOT, git=_git) -> UpdateStatus:
    """Fetch the current branch from origin and compare. Never raises for git / network
    trouble: the state says what happened."""
    root = Path(root)
    if not (root / ".git").exists():
        return UpdateStatus(NOT_GIT)
    try:
        r = git(["rev-parse", "--abbrev-ref", "HEAD"], root, 15)
    except FileNotFoundError:
        return UpdateStatus(NO_GIT_EXE)
    except subprocess.TimeoutExpired:
        return UpdateStatus(OFFLINE, detail="git rev-parse timed out")
    branch = r.stdout.strip()
    if r.returncode != 0 or not branch or branch == "HEAD":
        return UpdateStatus(DETACHED, detail=_out(r))
    try:
        r = git(["ls-remote", "--exit-code", "--heads", "origin", branch], root, 30)
        if r.returncode == 2:  # origin answered: no such branch
            return UpdateStatus(LOCAL_BRANCH, branch)
        if r.returncode != 0:
            return UpdateStatus(OFFLINE, branch, detail=_out(r))
        r = git(["-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=10", "fetch", "--quiet", "origin",
                 branch], root, 120)
        if r.returncode != 0:
            return UpdateStatus(OFFLINE, branch, detail=_out(r))
        r = git(["rev-list", "--count", f"HEAD..origin/{branch}"], root, 15)
        behind = int(r.stdout.strip() or 0) if r.returncode == 0 else 0
        if behind <= 0:
            return UpdateStatus(UP_TO_DATE, branch)
        r = git(["status", "--porcelain", "--untracked-files=no"], root, 15)
        if r.stdout.strip():
            return UpdateStatus(DIRTY, branch, behind)
        return UpdateStatus(BEHIND, branch, behind)
    except subprocess.TimeoutExpired as e:
        return UpdateStatus(OFFLINE, branch, detail=f"git {e.cmd[1] if e.cmd else ''} timed out")
    except ValueError as e:
        log.warning(f"Unexpected git output: {e}")
        return UpdateStatus(OFFLINE, branch, detail=str(e))


def pull(branch: str, root: Path = ROOT, git=_git):
    """Fast-forward to origin/branch. Returns (changed files, error text or None)."""
    root = Path(root)
    try:
        before = git(["rev-parse", "HEAD"], root, 15).stdout.strip()
        r = git(["pull", "--ff-only", "--quiet", "origin", branch], root, 300)
        if r.returncode != 0:
            return [], _out(r) or f"git pull exit code {r.returncode}"
        changed = git(["diff", "--name-only", before, "HEAD"], root, 15).stdout.split()
    except subprocess.TimeoutExpired:
        return [], "git pull timed out"
    except FileNotFoundError:
        return [], "git not found"
    log.info(f"Updated {branch}: {len(changed)} files changed")
    return changed, None


def packages_changed(changed) -> bool:
    return any(Path(f).name == "requirements.txt" for f in changed)


def launch_restart(parent_pid: int, packages: bool, root: Path = ROOT, popen=subprocess.Popen) -> bool:
    """Start launcher/update_window.ps1 (no console; its own small window): it waits for
    `parent_pid` to exit, reinstalls the packages when asked, and starts LiveTranslate."""
    if not UPDATE_WINDOW.exists():
        log.error(f"{UPDATE_WINDOW} not found; restart LiveTranslate yourself")
        return False
    args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-STA",
            "-File", str(UPDATE_WINDOW), "-ParentPid", str(parent_pid), "-Root", str(root)]
    if packages:
        args.append("-Packages")
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        popen(args, creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    except OSError as e:
        log.error(f"Cannot start the update window: {e}")
        return False
    return True
