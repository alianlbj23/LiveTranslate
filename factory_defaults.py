"""Factory (install-time) settings for the Japanese -> Taiwanese Mandarin setup.

Single source of the values written by launcher/write_settings.py (one-click install) and
restored by the control panel's 「回復原廠設定」. A factory reset puts every user-tunable
value back to these numbers but keeps what describes this machine and the user's
connections (KEEP_ON_RESET): the translation model list / Ollama connection, the ASR
model, audio devices and download settings.
"""
import copy

SYSTEM_PROMPT = (
    "你是即時字幕翻譯員，請把{source_lang}的直播對話翻譯成{target_lang}。\n"
    "規則：\n"
    "- 一律使用「台灣繁體中文」，用語習慣以台灣為準（例如：影片、網路、軟體、品質），絕對不要輸出簡體字。\n"
    "- 只輸出一個最佳翻譯，不要任何解釋、註解、括號或多個候選。\n"
    "- 人名、角色名、品牌名、遊戲與作品名保留原文或常見譯名。\n"
    "- 語氣口語自然，符合直播聊天情境，簡潔易讀。\n"
    "- 語音辨識可能有錯字，請依上下文與常識自行修正後再翻譯。\n"
    "- 若原文只是語助詞、笑聲或無意義的片段，輸出最貼近的簡短口語即可。\n"
    "\n"
    "範例（照這個風格翻）：\n"
    "えっ、マジで？やばすぎる → 欸，真的假的？太扯了吧\n"
    "スパチャありがとう！ → 謝謝超留！\n"
    "今日も配信見に来てくれてありがとう → 謝謝大家今天也來看直播"
)

# Same look as subtitle_overlay.DEFAULT_STYLE (translucent dark card), with Traditional Chinese fonts
STYLE = {
    "preset": "custom",
    "bg_color": "#16161d",
    "bg_opacity": 217,
    "header_color": "#ffffff",
    "header_opacity": 8,
    "border_radius": 12,
    "original_font_family": "Microsoft JhengHei",
    "translation_font_family": "Microsoft JhengHei",
    "original_font_size": 10,
    "translation_font_size": 15,
    "original_color": "#9a9aa8",
    "translation_color": "#ffffff",
    "timestamp_color": "#6c6c7c",
    "window_opacity": 100,
}

# Subtitle window: two sentences at once, each shown at least 0.3 s, so fast talkers keep up
SUBTITLE_MODE = {"sentences": 2, "min_display_ms": 300}

# Describe this machine / the user's connections: a factory reset keeps them
KEEP_ON_RESET = (
    "models",
    "active_model",
    "asr_engine",
    "whisper_model_size",
    "funasr_model",
    "asr_device",
    "remote_asr_url",
    "audio_device",
    "mic_device",
    "hub",
    "download_proxy",
    "cache_path",
)


def tunable_defaults() -> dict:
    """Every user-tunable value as the one-click installer writes it (fresh copy)."""
    return {
        "ui_lang": "zh_TW",
        "style": dict(STYLE),
        # --- ASR ---
        "asr_language": "ja",
        "whisper_pad_seconds": 0.5,
        "sensevoice_pad_seconds": 0.5,
        "incremental_asr": True,
        "interim_interval": 0.5,  # 每 0.5 秒做一次增量辨識（本 fork 把下限從 1.0 降到 0.3）
        "live_preview": True,  # 邊講邊翻：還沒講完的句子先顯示並暫時翻譯
        # --- VAD ---
        "vad_mode": "silero",
        "vad_threshold": 0.3,
        "energy_threshold": 0.02,
        "min_speech_duration": 0.6,
        "vad_min_speech_density": 0.25,  # 片段裡人聲比例低於 25% 就丟棄；調低可保留短回應
        "max_speech_duration": 5.0,
        "silence_mode": "fixed",  # 固定 0.5 秒停頓就送出，字幕出現得快；auto 最長會等 2 秒
        "silence_duration": 0.5,
        # --- translation ---
        "target_language": "zh",
        "timeout": 10,  # 翻譯總時間上限（秒）；5 秒在模型剛載入或排隊時太緊
        "auto_save_transcript": True,
        "system_prompt": SYSTEM_PROMPT,
        # --- subtitle window ---
        "subtitle_mode": dict(SUBTITLE_MODE),
    }


def install_settings(whisper_model: str, api_base: str, model_name: str) -> dict:
    """Everything launcher/write_settings.py writes on a fresh install."""
    s = tunable_defaults()
    s.update(
        {
            "hub": "hf",
            "download_proxy": "system",
            "audio_device": None,  # None = Windows 預設輸出裝置（WASAPI loopback）
            "mic_device": None,  # None = 不混入麥克風
            "asr_engine": "whisper",
            "whisper_model_size": whisper_model,
            "funasr_model": "sensevoice-small",
            "asr_device": "cuda:0",
            "models": [
                {
                    "name": f"Ollama {model_name}",
                    "api_base": api_base,
                    "api_key": "ollama",
                    "model": model_name,
                    "proxy": "none",
                    "thinking_style": "openai",  # Ollama /v1 only honours reasoning_effort=none
                    "streaming": True,
                    "context_turns": 3,
                }
            ],
            "active_model": 0,
        }
    )
    return s


def factory_reset(current) -> dict:
    """New settings with every tunable value back to the factory numbers.

    Keeps KEEP_ON_RESET and whether the subtitle window is open. Everything else, window
    positions included, is dropped so the app falls back to its defaults. `current`
    comes from user_settings.json, so anything that is not a dict is treated as empty.
    The result never shares objects with `current`."""
    cur = current if isinstance(current, dict) else {}
    new = tunable_defaults()
    for key in KEEP_ON_RESET:
        if key in cur:
            new[key] = copy.deepcopy(cur[key])
    if not isinstance(new.get("models", []), list):
        new.pop("models", None)
        new.pop("active_model", None)
    sm = cur.get("subtitle_mode")
    if isinstance(sm, dict) and isinstance(sm.get("enabled"), bool):
        new["subtitle_mode"]["enabled"] = sm["enabled"]
    return new
