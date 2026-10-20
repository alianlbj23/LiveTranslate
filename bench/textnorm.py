"""Japanese text normalisation and character error rate (CER).

Normalisation: NFKC (full-width ASCII -> half-width, half-width katakana -> full-width),
lower case, ASR special tags removed, then only letters and digits are kept (Unicode
categories L* and N*): punctuation, symbols and spaces go, while the long-vowel mark ー
(Lm) and the iteration mark 々 (Lm) stay. Hiragana and katakana are not unified, and
numbers are not converted (a SenseVoice "3" against a reference "三" counts as an error;
the report notes this)."""
import re
import unicodedata

_TAG_RE = re.compile(r"<\|[^|>]*\|>")


def normalize_ja(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _TAG_RE.sub("", text).lower()
    return "".join(c for c in text if unicodedata.category(c)[0] in ("L", "N"))


def edit_distance(ref: str, hyp: str) -> int:
    """Levenshtein distance between two strings (insert, delete, substitute = 1)."""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    previous = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        current = [i]
        for j, hc in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (rc != hc)))
        previous = current
    return previous[-1]


def clip_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance, reference length) after normalisation."""
    r, h = normalize_ja(ref), normalize_ja(hyp)
    return edit_distance(r, h), len(r)


def corpus_cer(pairs) -> float:
    """Total edit distance / total reference length over (ref, hyp) pairs. An empty
    hypothesis costs its whole reference."""
    dist = length = 0
    for ref, hyp in pairs:
        d, n = clip_errors(ref, hyp)
        dist += d
        length += n
    return dist / length if length else 0.0
