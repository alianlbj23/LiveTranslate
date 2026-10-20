"""Stage 3 translation: build a Translator exactly like main.py's _on_model_changed and
time one sentence (time to first token, full sentence, failure kind).

Context is off (set_context_turns(0)): the test sentences are unrelated to each other,
so earlier lines would only confuse the model (the report says so)."""
import logging
import time

log = logging.getLogger("LiveTranslate.Bench.TL")

WARMUP_SENTENCES = 3


def build_translator(entry: dict, settings: dict, config: dict, timeout: float | None = None):
    from translator import Translator

    tl_cfg = config.get("translation", {})
    prompt = settings.get("system_prompt") or tl_cfg.get("system_prompt")
    translator = Translator(
        api_base=entry["api_base"],
        api_key=entry["api_key"],
        model=entry["model"],
        target_language=settings.get("target_language", tl_cfg.get("target_language", "zh")),
        max_tokens=tl_cfg.get("max_tokens", 256),
        temperature=tl_cfg.get("temperature", 0.3),
        streaming=entry.get("streaming", True),
        system_prompt=prompt,
        proxy=entry.get("proxy", "none"),
        no_system_role=entry.get("no_system_role", False),
        no_think=entry.get("no_think", True),
        thinking_style=entry.get("thinking_style"),
        json_response=entry.get("json_response", False),
        timeout=timeout if timeout is not None else settings.get("timeout", 10),
        overrides=entry.get("overrides"),
        extra_body=entry.get("extra_body"),
        zh_tw_postprocess=tl_cfg.get("zh_tw_postprocess", False),
        prompt_style=entry.get("prompt_style"),
    )
    translator.set_context_turns(0)
    return translator


def _kana_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    kana = sum(1 for c in letters if "぀" <= c <= "ヿ")
    return kana / len(letters)


def classify_error(exc: Exception) -> str:
    import httpx
    import openai
    from translator import RepetitionError

    if isinstance(exc, RepetitionError):
        return "repetition"
    if isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException, TimeoutError)):
        return "timeout"
    if isinstance(exc, (openai.APIConnectionError, httpx.ConnectError, ConnectionError)):
        return "connection"
    if isinstance(exc, openai.APIStatusError):
        return f"http_{exc.status_code}"
    return type(exc).__name__


def translate_timed(translator, text: str, source_language: str = "ja", clock=time.perf_counter) -> dict:
    """{"ok", "text", "ttft_ms", "total_ms", "error"}. Kinds of failure: an exception
    (timeout / connection / http_N / repetition / ...), "empty" output, or "untranslated"
    (mostly kana left: the model echoed Japanese)."""
    t0 = clock()
    ttft = None
    final = ""
    gen = translator.translate_iter(text, source_language)
    try:
        for partial in gen:
            if ttft is None and partial:
                ttft = (clock() - t0) * 1000
            final = partial
    except Exception as e:  # recorded as a failed sentence, the run goes on
        return {"ok": False, "text": final, "ttft_ms": ttft, "total_ms": (clock() - t0) * 1000,
                "error": classify_error(e), "detail": str(e)[:200]}
    finally:
        gen.close()
    total = (clock() - t0) * 1000
    final = (final or "").strip()
    if not final:
        return {"ok": False, "text": "", "ttft_ms": ttft, "total_ms": total, "error": "empty"}
    if _kana_ratio(final) > 0.3:
        return {"ok": False, "text": final, "ttft_ms": ttft, "total_ms": total, "error": "untranslated"}
    return {"ok": True, "text": final, "ttft_ms": ttft if ttft is not None else total, "total_ms": total,
            "error": None}


def availability(entry: dict, ollama_installed: set | None, lmstudio_alive: bool | None) -> tuple[bool, str]:
    """(usable, reason shown in the menu)."""
    from bench.llm_servers import model_kind

    kind = model_kind(entry)
    if kind == "ollama":
        if ollama_installed is None:
            return False, "Ollama 沒有回應"
        return (entry.get("model") in ollama_installed,
                "已下載" if entry.get("model") in ollama_installed else "尚未下載")
    if kind == "lmstudio":
        return (bool(lmstudio_alive), "LM Studio 執行中" if lmstudio_alive else "LM Studio 沒有開")
    if kind == "cloud":
        return (bool(entry.get("api_key")), "雲端" if entry.get("api_key") else "沒有 API key")
    return True, "本機伺服器"
