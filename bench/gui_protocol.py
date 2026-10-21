"""JSON-lines protocol between the benchmark window (bench/window.py) and the benchmark
running in a child process (launcher/bench_run.py <command> --json).

The child writes one JSON object per line to stdout. Prompts wait for one JSON answer line
on stdin; an empty stdin (the window went away) answers like cancelling. Progress the
benchmark rewrites in place with "\\r" (downloads, ProgressLine) becomes "progress"
messages. Messages:

  child → window   {"t": "say", "lines": [...]}
                   {"t": "progress", "text": "..."}            ("" clears it)
                   {"t": "ask", "prompt": "..."}               → {"text": "..."}
                   {"t": "confirm", "q": "...", "default": b}  → {"yes": b}
                   {"t": "pause", "message": "..."}            → {}
                   {"t": "choose_one", "title", "labels", "default", "cancel"}
                                                               → {"index": i | null}
                   {"t": "choose_many", "title", "items": [{"label", "enabled", "note", "checked"}],
                    "minimum"}                                 → {"indexes": [...] | null}"""
import json

from bench.wizard import Console


class JsonConsole(Console):
    def __init__(self, out, inp):
        super().__init__()
        self._out = out
        self._in = inp

    def send(self, msg: dict):
        self._out.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._out.flush()

    def _request(self, msg: dict):
        """Send a prompt and read the answer; None when the window is gone or answered garbage."""
        self.send(msg)
        line = self._in.readline()
        if not line:
            return None
        try:
            answer = json.loads(line)
        except ValueError:
            return None
        return answer if isinstance(answer, dict) else None

    def say(self, *lines):
        self.send({"t": "say", "lines": [str(line) for line in lines]})

    def ask(self, prompt: str) -> str:
        answer = self._request({"t": "ask", "prompt": prompt})
        return "0" if answer is None else str(answer.get("text", "")).strip()

    def confirm(self, question: str, default: bool = True) -> bool:
        answer = self._request({"t": "confirm", "q": question, "default": bool(default)})
        return False if answer is None else bool(answer.get("yes", default))

    def pause(self, message: str = "按 Enter 繼續"):
        self._request({"t": "pause", "message": message})

    def choose_one(self, title: str, options: list, default: int = 0, allow_cancel: bool = True):
        if not options:
            self.say("", "沒有可以選的項目。")
            return None
        answer = self._request({"t": "choose_one", "title": title, "labels": [str(o[1]) for o in options],
                                "default": default, "cancel": allow_cancel})
        index = None if answer is None else answer.get("index")
        if isinstance(index, int) and 0 <= index < len(options):
            return options[index][0]
        return None if allow_cancel else options[default][0]

    def choose_many(self, title: str, options: list, selected: set, minimum: int = 1):
        selected = {k for k in selected if any(o[0] == k and o[2] for o in options)}
        items = [{"label": str(label), "enabled": bool(enabled), "note": str(note or ""), "checked": key in selected}
                 for key, label, enabled, note in options]
        while True:
            answer = self._request({"t": "choose_many", "title": title, "items": items, "minimum": minimum})
            indexes = None if answer is None else answer.get("indexes")
            if indexes is None:
                return None
            picked = [i for i in indexes if isinstance(i, int) and 0 <= i < len(options) and options[i][2]]
            if len(picked) >= minimum and len(picked) == len(indexes):
                return [options[i][0] for i in sorted(set(picked))]
            self.say(f"至少要勾選 {minimum} 項，而且不能勾選停用的項目。")


class ProgressStream:
    """Stands in for sys.stdout in the child: text rewritten with "\\r" becomes progress
    messages, finished lines become "say" messages."""

    encoding = "utf-8"

    def __init__(self, send):
        self._send = send
        self._buf = ""
        self._progress_shown = False

    def write(self, s: str) -> int:
        for ch in s:
            if ch == "\r":
                self._buf = ""
            elif ch == "\n":
                line, self._buf = self._buf, ""
                if self._progress_shown:
                    self._send({"t": "progress", "text": ""})
                    self._progress_shown = False
                self._send({"t": "say", "lines": [line]})
            else:
                self._buf += ch
        return len(s)

    def flush(self):
        if self._buf:
            self._send({"t": "progress", "text": self._buf.strip()})
            self._progress_shown = True

    def isatty(self) -> bool:
        return False
