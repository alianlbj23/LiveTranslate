"""Update check / update (Settings → Home → Maintenance, plus a quiet check after start).

Two kinds of installation:
- git checkout: fetch the current branch and fast-forward (replaces the old
  launcher/update.ps1; skipped while tracked files have local changes).
- release package: VERSION.json (written by build_release.ps1) names the version and the
  GitHub repository; the latest release's zip is downloaded, checked and copied over the
  program files after LiveTranslate has closed (.venv, models, settings and logs are not in
  the zip, so they stay).

Either way launcher/update_window.ps1 finishes the job once LiveTranslate has closed (pip
cannot replace the DLLs a running app holds): it copies a downloaded release, reinstalls the
packages when requirements.txt changed and starts LiveTranslate again. No Qt."""
import json
import logging
import os
import re
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("LiveTranslate.Update")

ROOT = Path(__file__).resolve().parent
UPDATE_WINDOW = ROOT / "launcher" / "update_window.ps1"
VERSION_FILE = "VERSION.json"
DOWNLOAD_DIR = ".update"
MAX_DOWNLOAD_BYTES = 500 * 2**20   # a release zip is about 1 MB; anything this big is not one
MAX_UNPACKED_BYTES = 2 * 2**30     # what its files may take once unpacked (a zip bomb would not stop)
DOWNLOAD_DEADLINE_S = 30 * 60      # the whole download, however slowly the bytes trickle in
DOWNLOAD_HOSTS = ("github.com", ".githubusercontent.com")  # release downloads redirect to GitHub's CDN
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_ZIP_RE = re.compile(r"^LiveTranslate-portable-[A-Za-z0-9._-]+\.zip$")

# states
NOT_GIT = "not_git"            # neither a git checkout nor a release package with VERSION.json
NO_GIT_EXE = "no_git_exe"      # git is not installed
DETACHED = "detached"          # not on a branch
LOCAL_BRANCH = "local_branch"  # the branch exists only on this computer
OFFLINE = "offline"            # GitHub could not be reached
UP_TO_DATE = "up_to_date"
BEHIND = "behind"              # newer commits on GitHub, can update
DIRTY = "dirty"                # newer commits, but tracked files were edited here
RELEASE_UP_TO_DATE = "release_up_to_date"
RELEASE_NEWER = "release_newer"  # a newer release on GitHub, can update
NO_RELEASE = "no_release"        # the repository has no release (or does not exist)


@dataclass
class UpdateStatus:
    state: str
    branch: str = ""
    behind: int = 0
    detail: str = ""
    version: str = ""     # release package: the installed version
    latest: str = ""      # release package: the newest release
    asset_url: str = ""   # release package: its zip

    @property
    def can_update(self) -> bool:
        return self.state in (BEHIND, RELEASE_NEWER)

    @property
    def is_release(self) -> bool:
        """A release package (it knows its installed version)."""
        return bool(self.version)


def _git(args, root: Path, timeout: float) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")  # never wait for a password prompt
    return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=env,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _out(r: subprocess.CompletedProcess) -> str:
    return (r.stderr or r.stdout or "").strip()[:300]


def check(root: Path = ROOT, git=_git, http=None) -> UpdateStatus:
    """What kind of installation this is and whether GitHub has something newer. Never
    raises for git / network trouble: the state says what happened."""
    root = Path(root)
    if (root / ".git").exists():
        return _check_git(root, git)
    info = read_version(root)
    if info is None:
        return UpdateStatus(NOT_GIT)
    return check_release(info, http)


def _check_git(root: Path, git) -> UpdateStatus:
    """Fetch the current branch from origin and compare."""
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
        r = git(["rev-parse", "HEAD"], root, 15)
        before = r.stdout.strip()
        if r.returncode != 0 or not before:  # without it the changed files (requirements.txt) are unknown
            return [], _out(r) or "git rev-parse HEAD failed"
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


# ---------------------------------------------------------------- release packages

def read_version(root: Path = ROOT):
    """{"version", "repo"} from VERSION.json of a release package, or None (no file, or one
    that does not look like ours: then no update is offered)."""
    path = Path(root) / VERSION_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        log.warning(f"Cannot read {path}: {e}")
        return None
    version = data.get("version") if isinstance(data, dict) else None
    repo = data.get("repo") if isinstance(data, dict) else None
    if not (isinstance(version, str) and version.strip() and isinstance(repo, str) and _REPO_RE.match(repo)):
        log.warning(f"{path} has no usable version / repo: {data!r}")
        return None
    return {"version": version.strip(), "repo": repo}


def _numbers(tag: str):
    """v2026.10.21 -> (2026, 10, 21); None for tags that are not dotted numbers."""
    m = re.fullmatch(r"v?(\d+(?:\.\d+)*)", tag.strip())
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def is_newer(latest: str, current: str) -> bool:
    """Only dotted-number tags (v2026.10.22) are compared; anything else is never offered, so
    an oddly named release can not downgrade an installation."""
    a, b = _numbers(latest), _numbers(current)
    return a is not None and b is not None and a > b


def _client(http):
    import httpx

    return http or httpx.Client(timeout=httpx.Timeout(20.0, connect=10.0), follow_redirects=True,
                                headers={"Accept": "application/vnd.github+json",
                                         "User-Agent": "LiveTranslate-updater"})


def check_release(info: dict, http=None) -> UpdateStatus:
    """Compare the installed version with the repository's latest GitHub release."""
    import httpx

    version, repo = info["version"], info["repo"]
    try:
        resp = _client(http).get(f"https://api.github.com/repos/{repo}/releases/latest")
        if resp.status_code == 404:  # no release published yet, or no such repository
            return UpdateStatus(NO_RELEASE, version=version, detail=f"no release in {repo}")
        if resp.status_code != 200:
            return UpdateStatus(OFFLINE, version=version, detail=f"GitHub HTTP {resp.status_code}")
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        return UpdateStatus(OFFLINE, version=version, detail=f"{type(e).__name__}: {e}")
    tag = data.get("tag_name") if isinstance(data, dict) else None
    if not isinstance(tag, str) or not tag.strip():
        return UpdateStatus(OFFLINE, version=version, detail="the latest release has no tag")
    if not is_newer(tag, version):
        return UpdateStatus(RELEASE_UP_TO_DATE, version=version, latest=tag)
    prefix = f"https://github.com/{repo}/releases/download/".lower()  # GitHub names are case-insensitive
    assets = data.get("assets") if isinstance(data.get("assets"), list) else []
    url = next((a.get("browser_download_url") for a in assets
                if isinstance(a, dict) and _ZIP_RE.match(str(a.get("name", "")))
                and str(a.get("browser_download_url", "")).lower().startswith(prefix)), None)
    if not url:
        return UpdateStatus(OFFLINE, version=version, latest=tag, detail=f"release {tag} has no package zip")
    return UpdateStatus(RELEASE_NEWER, version=version, latest=tag, asset_url=url)


def _check_zip(path: Path):
    """None when `path` is a complete release zip, else what is wrong."""
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            bad = z.testzip()
    except (OSError, zipfile.BadZipFile) as e:
        return f"not a zip: {e}"
    if bad:
        return f"damaged file in the zip: {bad}"
    names = [i.filename.replace("\\", "/") for i in infos]
    for name in names:
        # Windows drops trailing spaces / dots of a path part (".. " is ".."), so judge it that way
        parts = [part.rstrip(" .") if part not in (".", "..") else part for part in name.split("/")]
        if (parts[0] != "LiveTranslate" or any(part in (".", "..") for part in parts)
                or ":" in name or name.startswith("/") or any(not part for part in parts[:-1])):
            return f"unexpected path in the zip: {name}"
    if sum(i.file_size for i in infos) > MAX_UNPACKED_BYTES:
        return "the zip unpacks to more than expected"
    for need in ("LiveTranslate/main.py", "LiveTranslate/LiveTranslate.exe", f"LiveTranslate/{VERSION_FILE}"):
        if need not in names:
            return f"{need} is missing"
    return None


def download_release(status: UpdateStatus, root: Path = ROOT, http=None, on_progress=None):
    """Download the release zip into <root>/.update and check it. Returns (path, error)."""
    url = status.asset_url
    if not url.startswith("https://github.com/") or "/releases/download/" not in url:
        return None, f"refusing to download from {url!r}"
    folder = Path(root) / DOWNLOAD_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "LiveTranslate-update.zip"
    error = _download(url, path, http, on_progress)
    if error is None:
        error = _check_zip(path)
    if error:
        log.error(f"Release {status.latest} not downloaded: {error}")
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            log.warning(f"Could not remove {path}: {e}")
        return None, error
    log.info(f"Downloaded release {status.latest} to {path}")
    return path, None


def _download(url: str, path: Path, http, on_progress):
    """Stream `url` into `path`; None, or what went wrong."""
    import httpx

    client = _client(http)
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_S
    done = 0
    try:
        with client.stream("GET", url) as resp:
            host = (resp.url.host or "").lower()
            if not (host == DOWNLOAD_HOSTS[0] or host.endswith(DOWNLOAD_HOSTS[1])):
                return f"refusing a download redirected to {host}"
            if resp.status_code != 200:
                return f"HTTP {resp.status_code}"
            total = int(resp.headers.get("Content-Length") or 0)
            if total > MAX_DOWNLOAD_BYTES:
                return f"the download is too big ({total} bytes)"
            with open(path, "wb") as f:
                for chunk in resp.iter_bytes():
                    done += len(chunk)
                    if done > MAX_DOWNLOAD_BYTES:
                        return "the download is too big"
                    if time.monotonic() > deadline:
                        return f"the download took longer than {DOWNLOAD_DEADLINE_S // 60} minutes"
                    f.write(chunk)
                    if on_progress is not None:
                        on_progress(done, total)
    except (httpx.HTTPError, OSError, ValueError) as e:
        return f"{type(e).__name__}: {e}"
    finally:
        if http is None:
            client.close()
    return None


def release_packages_changed(zip_path: Path, root: Path = ROOT) -> bool:
    """True when the release's requirements.txt differs from the installed one."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            new = z.read("LiveTranslate/requirements.txt")
    except KeyError:
        return False
    except (OSError, zipfile.BadZipFile) as e:
        log.warning(f"Cannot read requirements.txt from {zip_path}: {e}; reinstalling the packages")
        return True
    try:
        old = (Path(root) / "requirements.txt").read_bytes()
    except OSError:
        return True
    return new.replace(b"\r\n", b"\n") != old.replace(b"\r\n", b"\n")


def launch_restart(parent_pid: int, reinstall_packages: bool, root: Path = ROOT, popen=subprocess.Popen,
                   release_zip: Path | None = None, started_timeout: float = 30.0, sleep=time.sleep) -> bool:
    """Start launcher/update_window.ps1 (no console; its own small window): it waits for
    `parent_pid` to exit, copies `release_zip` over the program files when given, reinstalls
    the packages when asked, and starts LiveTranslate. True only once the window reported that
    it runs (it writes a file): LiveTranslate must not quit for a window that never came up
    (e.g. a policy that blocks scripts); its errors go to logs\\update_launch.log."""
    if not UPDATE_WINDOW.exists():
        log.error(f"{UPDATE_WINDOW} not found; restart LiveTranslate yourself")
        return False
    root = Path(root)
    started = root / DOWNLOAD_DIR / f"window_started_{parent_pid}"
    started.parent.mkdir(parents=True, exist_ok=True)
    try:
        started.unlink()
    except FileNotFoundError:
        pass
    args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-STA",
            "-File", str(UPDATE_WINDOW), "-ParentPid", str(parent_pid), "-Root", str(root),
            "-StartedFile", str(started)]
    if release_zip is not None:
        args += ["-Zip", str(release_zip)]
    if reinstall_packages:
        args.append("-Packages")
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        with open(logs / "update_launch.log", "ab") as err:
            proc = popen(args, creationflags=flags, stdout=subprocess.DEVNULL, stderr=err, close_fds=True)
    except OSError as e:
        log.error(f"Cannot start the update window: {e}")
        return False
    deadline = time.monotonic() + started_timeout
    while time.monotonic() < deadline:
        if started.exists():
            try:
                started.unlink()
            except OSError as e:
                log.info(f"Could not remove {started}: {e}")
            return True
        if proc is not None and proc.poll() is not None:
            log.error(f"The update window ended at once (exit code {proc.returncode}); see logs\\update_launch.log")
            return False
        sleep(0.2)
    log.error(f"The update window did not come up within {started_timeout:.0f} s")
    return False
