"""Token and API cost estimates shown before a run starts (local models cost 0).

Tokens: about 1 per CJK character and 1 per 4 ASCII characters. Prices come from the
model entry (input_price / output_price per 1M tokens); a cloud model without prices is
reported as "price not set" instead of a guess. The range shown is +-30 %."""

JUDGE_RULES_TOKENS = 600
JUDGE_OUTPUT_TOKENS = 60
VAD_RULES_TOKENS = 350
VAD_OUTPUT_TOKENS = 80


def est_tokens(text: str) -> int:
    text = text or ""
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (len(text) - ascii_chars) + (ascii_chars + 3) // 4


def call_cost(entry: dict, tokens_in: int, tokens_out: int):
    """US$ for one model, or None when a cloud model has no prices set."""
    from bench.llm_servers import model_kind

    if model_kind(entry) != "cloud":
        return 0.0
    pin, pout = entry.get("input_price"), entry.get("output_price")
    if not pin and not pout:
        return None
    return (tokens_in * float(pin or 0) + tokens_out * float(pout or 0)) / 1_000_000


def translation_tokens(system_prompt: str, sentences: list) -> tuple[int, int]:
    base = est_tokens(system_prompt or "")
    tin = sum(base + est_tokens(s) for s in sentences)
    tout = sum(int(1.3 * est_tokens(s)) + 2 for s in sentences)
    return tin, tout


def judge_tokens(n_calls: int, avg_sentence_tokens: int = 25) -> tuple[int, int]:
    return n_calls * (JUDGE_RULES_TOKENS + 2 * avg_sentence_tokens), n_calls * JUDGE_OUTPUT_TOKENS


def vad_judge_tokens(n_windows: int, window_tokens: int = 250) -> tuple[int, int]:
    return n_windows * (VAD_RULES_TOKENS + 2 * window_tokens), n_windows * VAD_OUTPUT_TOKENS


def fmt_cost(value) -> str:
    if value is None:
        return "未設定價格"
    if value == 0:
        return "US$0（本機模型）"
    return f"約 US${value * 0.7:.2f}–{value * 1.3:.2f}"
