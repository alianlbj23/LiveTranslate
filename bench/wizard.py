"""Console prompts for the benchmark menu: numbered single / multi choice and yes / no.
Input and output are injectable so the flows can be tested without a terminal."""


class Console:
    def __init__(self, input_fn=input, print_fn=print):
        self._input = input_fn
        self._print = print_fn

    def say(self, *lines):
        for line in lines:
            self._print(line)

    def ask(self, prompt: str) -> str:
        try:
            return (self._input(prompt) or "").strip()
        except EOFError:
            return "0"

    def confirm(self, question: str, default: bool = True) -> bool:
        hint = "(Y/n)" if default else "(y/N)"
        while True:
            answer = self.ask(f"{question} {hint} ").lower()
            if not answer:
                return default
            if answer in ("y", "yes", "是"):
                return True
            if answer in ("n", "no", "否", "0"):
                return False
            self.say("請輸入 y 或 n。")

    def pause(self, message: str = "按 Enter 繼續"):
        self.ask(message)

    def choose_one(self, title: str, options: list, default: int = 0, allow_cancel: bool = True):
        """options: [(key, label)]. Returns the key, or None when cancelled with 0 (or
        when there is nothing to choose)."""
        if not options:
            self.say("", "沒有可以選的項目。")
            return None
        self.say("", title)
        for i, (_key, label) in enumerate(options, 1):
            mark = "  （預設）" if i - 1 == default else ""
            self.say(f" {i}) {label}{mark}")
        if allow_cancel:
            self.say(" 0) 取消")
        while True:
            answer = self.ask("請輸入數字（直接按 Enter = 預設）：")
            if not answer:
                return options[default][0]
            if answer == "0" and allow_cancel:
                return None
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return options[int(answer) - 1][0]
            self.say(f"請輸入 1–{len(options)} 的數字。")

    def choose_many(self, title: str, options: list, selected: set, minimum: int = 1):
        """options: [(key, label, enabled, note)]. Typing a number toggles [x] / [ ];
        Enter confirms. Disabled options cannot be ticked. Returns the selected keys, or
        None when cancelled with 0."""
        selected = {k for k in selected if any(o[0] == k and o[2] for o in options)}
        while True:
            self.say("", title)
            for i, (key, label, enabled, note) in enumerate(options, 1):
                box = "[x]" if key in selected else ("[-]" if not enabled else "[ ]")
                self.say(f" {i}) {box} {label}" + (f"  — {note}" if note else ""))
            self.say(" 0) 取消")
            answer = self.ask("輸入數字切換勾選，直接按 Enter 確定：")
            if not answer:
                if len(selected) >= minimum:
                    return [o[0] for o in options if o[0] in selected]
                self.say(f"至少要勾選 {minimum} 項。")
                continue
            if answer == "0":
                return None
            picks = [p for p in answer.replace("，", ",").replace(" ", ",").split(",") if p]
            for p in picks:
                if not (p.isdigit() and 1 <= int(p) <= len(options)):
                    self.say(f"「{p}」不是有效的編號。")
                    continue
                key, _label, enabled, note = options[int(p) - 1]
                if not enabled:
                    self.say(f"這一項不能勾選：{note}")
                    continue
                selected.symmetric_difference_update({key})
