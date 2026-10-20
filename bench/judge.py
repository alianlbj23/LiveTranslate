"""LLM judging.

Stage 3: blind 1-10 scores on four criteria against the Japanese reference.
Cross judging: a candidate is never scored by itself when another judge exists; judges
are aligned with per-judge offsets computed on "anchor" items every judge scored (the
calibration set and candidates scored by several judges).
Stage 2: hallucinated / missed subtitle lines against a continuous transcript."""
import json
import logging
import re
from statistics import mean

log = logging.getLogger("LiveTranslate.Bench.Judge")

CRITERIA = ("accuracy", "naturalness", "tone", "taiwan")
CRITERIA_LABELS = {"accuracy": "準確度", "naturalness": "自然度", "tone": "語氣/角色感", "taiwan": "台灣用語"}

TL_SYSTEM = (
    "你是日文→台灣繁體中文字幕評審。以「日文原文（正確答案）」的意思為準；譯文來自語音辨識結果，"
    "辨識錯誤造成的翻譯錯誤也算譯文錯。\n"
    "四項各給 1–10 的整數：\n"
    "accuracy 意思正確（沒有漏譯、誤譯或多加內容）\n"
    "naturalness 自然流暢（像台灣字幕組的字幕）\n"
    "tone 語氣與角色感（口吻、情緒、敬語、性別、角色特色）\n"
    "taiwan 台灣用語（正體字、台灣慣用詞，沒有簡體字或中國用語）\n"
    "基準：10 專業字幕；8 小瑕疵；6 看得懂但不自然或小錯；4 部分意思錯；2 大部分錯；1 全錯、空白或沒有翻譯。\n"
    '只輸出一行 JSON，例如 {"accuracy":8,"naturalness":7,"tone":6,"taiwan":9,"note":"20 字內主要問題"}'
)
CATEGORY_LABELS = {"anime": "動畫台詞", "vtuber": "直播口語"}

# (id, Japanese, translation, expectation): expectation = {criterion: (low, high)}
CALIBRATION = [
    ("cal-good", "今日は本当にありがとうございました。", "今天真的非常謝謝你。", {"accuracy": (8, 10)}),
    ("cal-good2", "お前、まだそんなこと言ってるのか？", "你還在說那種話嗎？", {"accuracy": (8, 10)}),
    ("cal-untranslated", "明日は雨が降るらしいよ。", "明日は雨が降るらしいよ。", {"accuracy": (1, 3)}),
    ("cal-simplified", "このソフトウェアの品質はとても高いです。", "这个软件的质量非常高。", {"taiwan": (1, 5)}),
    ("cal-opposite", "絶対に行かないよ。", "我一定會去的！", {"accuracy": (1, 3)}),
    ("cal-stiff", "ちょっと待ってよ、置いていかないで！", "稍微等待一下吧，不要把我放置後前往！", {"naturalness": (1, 6)}),
]

VAD_SYSTEM = (
    "你要檢查即時字幕的斷句品質。給你一段日文直播的「連續稿」（同一段聲音不經切段直接辨識的結果，視為參考）"
    "和同一段時間內的「字幕行」（經過語音偵測切段後辨識，前面有行號）。\n"
    "找出：hallucinated = 連續稿裡沒有對應內容的字幕行行號（幻聽）；"
    "missed = 連續稿裡有、字幕行完全沒有的句子（簡短引用，每句 20 字內）。意思相同但用字略不同不算。\n"
    '只輸出一行 JSON：{"hallucinated":[行號...],"missed":["..."]}'
)

_JSON_RE = re.compile(r"\{.*?\}", re.S)


def parse_scores(text: str):
    """First {...} with the four criteria as numbers 0..10 -> ints 1..10, or None. A
    value outside 0..10 (e.g. a 100-point scale) means the rubric was not followed."""
    for m in _JSON_RE.finditer(text or ""):
        try:
            data = json.loads(m.group(0))
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        try:
            values = {c: float(data[c]) for c in CRITERIA}
        except (KeyError, TypeError, ValueError):
            continue
        if any(not (0 <= v <= 10) for v in values.values()):
            log.warning(f"Judge scores out of the 1-10 range: {values}")
            return None
        scores = {c: min(10, max(1, int(round(v)))) for c, v in values.items()}
        scores["note"] = str(data.get("note", ""))[:40]
        return scores
    return None


def parse_vad(text: str, n_lines: int):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    lines = data.get("hallucinated", [])
    missed = data.get("missed", [])
    if not isinstance(lines, list) or not isinstance(missed, list):
        return None
    numbers = []
    for x in lines:
        try:
            numbers.append(int(float(x)))
        except (TypeError, ValueError):
            log.debug(f"Ignoring a non-numeric line number from the VAD judge: {x!r}")
    hall = sorted({n for n in numbers if 1 <= n <= n_lines})
    return {"hallucinated": hall, "missed": [str(x)[:40] for x in missed if str(x).strip()]}


def tl_user_prompt(category: str, ja: str, translation: str) -> str:
    return f"類別：{CATEGORY_LABELS.get(category, category)}\n日文原文：{ja}\n待評譯文：{translation}"


def vad_user_prompt(continuous: str, lines: list) -> str:
    body = "\n".join(f"{i}. {t}" for i, t in enumerate(lines, 1)) or "（沒有字幕行）"
    return f"連續稿：\n{continuous}\n\n字幕行：\n{body}"


class JudgeClient:
    """OpenAI-compatible chat call with thinking disabled the same way the translator
    does it (thinking_style), temperature 0 and a fixed seed."""

    def __init__(self, entry: dict, timeout: float = 60.0, client=None):
        from translator import make_openai_client, resolve_thinking_style, thinking_disable_body

        self.entry = entry
        if client is None:
            # no hidden retries: a failing judge is counted and stopped by the orchestrator
            client = make_openai_client(entry["api_base"], entry.get("api_key") or "none",
                                        entry.get("proxy", "none"), timeout=timeout).with_options(max_retries=0)
        self._client = client
        body = dict(thinking_disable_body(resolve_thinking_style(entry.get("thinking_style"),
                                                                 entry["api_base"], entry["model"])))
        body.update(entry.get("extra_body") or {})
        self._extra_body = body
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def ask(self, system: str, user: str, max_tokens: int = 200) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.entry.get("no_system_role"):
            messages = [{"role": "user", "content": system + "\n\n" + user}]
        resp = self._client.chat.completions.create(
            model=self.entry["model"], messages=messages, temperature=0, seed=1234,
            max_tokens=max_tokens, extra_body=self._extra_body or None)
        if getattr(resp, "usage", None):
            self.prompt_tokens += resp.usage.prompt_tokens or 0
            self.completion_tokens += resp.usage.completion_tokens or 0
        return (resp.choices[0].message.content or "").strip()

    def score(self, category: str, ja: str, translation: str):
        """Scores dict, or None after one retry (judge_fail)."""
        user = tl_user_prompt(category, ja, translation)
        for extra in ("", "\n只輸出 JSON，不要其他文字。"):
            try:
                answer = self.ask(TL_SYSTEM, user + extra)
            except Exception as e:  # a failed judge call is counted as judge_fail
                log.warning(f"Judge {self.entry.get('name')} call failed: {e}")
                continue
            scores = parse_scores(answer)
            if scores:
                return scores
            log.warning(f"Judge {self.entry.get('name')} answer is not the JSON asked for: {answer[:120]!r}")
        return None

    def vad(self, continuous: str, lines: list):
        user = vad_user_prompt(continuous, lines)
        for extra in ("", "\n只輸出 JSON，不要其他文字。"):
            try:
                result = parse_vad(self.ask(VAD_SYSTEM, user + extra, max_tokens=400), len(lines))
            except Exception as e:  # counted as a failed window
                log.warning(f"VAD judge call failed: {e}")
                result = None
            if result is not None:
                return result
        return None


def eligible_judge(entry: dict) -> bool:
    """Sakura-style models only translate; they cannot follow a scoring rubric."""
    return entry.get("prompt_style") != "sakura"


def assign_judges(candidates: list, judges: list) -> dict:
    """candidate fingerprint -> [(judge fingerprint, is_self)]. Never self-judged when
    another judge exists."""
    out = {}
    for c in candidates:
        others = [j for j in judges if j != c]
        out[c] = [(j, False) for j in others] if others else [(j, True) for j in judges if j == c]
    return out


def item_mean(scores: dict) -> float:
    return mean(scores[c] for c in CRITERIA)


def judge_offsets(records: list) -> dict:
    """Per-judge offset: mean over anchor items (scored by >= 2 judges) of the judge's
    item mean minus the average of all judges on that item. 0 without anchors."""
    by_item = {}
    for r in records:
        if r.get("scores"):
            by_item.setdefault(r["item"], {})[r["judge"]] = item_mean(r["scores"])
    diffs = {}
    for scores in by_item.values():
        if len(scores) < 2:
            continue
        avg = mean(scores.values())
        for judge, s in scores.items():
            diffs.setdefault(judge, []).append(s - avg)
    judges = {r["judge"] for r in records}
    return {j: (mean(diffs[j]) if diffs.get(j) else 0.0) for j in judges}


def calibration_report(records: list) -> dict:
    """judge -> {"misses": n, "items": n}; >= 2 misses flags the judge as unreliable."""
    expect = {cid: exp for cid, _, _, exp in CALIBRATION}
    out = {}
    for r in records:
        cid = r["item"].split("/", 1)[-1]
        if cid not in expect:
            continue
        stat = out.setdefault(r["judge"], {"misses": 0, "items": 0})
        stat["items"] += 1
        scores = r.get("scores")
        if not scores or any(not (lo <= scores[c] <= hi) for c, (lo, hi) in expect[cid].items()):
            stat["misses"] += 1
    return out
