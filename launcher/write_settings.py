"""Write the initial user_settings.json for the Japanese -> Taiwanese Mandarin setup.

Usage: python -I write_settings.py <project_root> <whisper_model_path_or_size> <api_base> <model_name>

Existing user_settings.json is merged (our keys win) so overlay positions etc. survive.
"""
import json
import os
import sys

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

STYLE = {
    "preset": "custom",
    "bg_color": "#000000",
    "bg_opacity": 240,
    "header_color": "#1a1a2e",
    "header_opacity": 230,
    "border_radius": 8,
    "original_font_family": "Microsoft JhengHei",
    "translation_font_family": "Microsoft JhengHei",
    "original_font_size": 11,
    "translation_font_size": 15,
    "original_color": "#cccccc",
    "translation_color": "#ffffff",
    "timestamp_color": "#888899",
    "window_opacity": 95,
}


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__)
        return 2
    root, whisper_model, api_base, model_name = sys.argv[1:5]
    path = os.path.join(root, "user_settings.json")

    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    data.update(
        {
            "hub": "hf",
            "download_proxy": "system",
            "ui_lang": "zh_TW",
            # 字幕字型：JhengHei 才有台灣慣用字形
            "style": STYLE,
            # --- audio ---
            "audio_device": None,  # None = Windows 預設輸出裝置（WASAPI loopback）
            "mic_device": None,  # None = 不混入麥克風
            # --- ASR ---
            "asr_engine": "whisper",
            "whisper_model_size": whisper_model,
            "funasr_model": "sensevoice-small",
            "asr_language": "ja",
            "asr_device": "cuda:0",
            "whisper_pad_seconds": 0.5,
            "incremental_asr": True,
            "interim_interval": 1.0,  # GUI 下限就是 1.0 秒，管線另有 1 秒冷卻
            # --- VAD ---
            "vad_mode": "silero",
            "vad_threshold": 0.3,
            "energy_threshold": 0.02,
            "min_speech_duration": 0.6,
            "max_speech_duration": 5.0,
            "silence_mode": "fixed",  # 固定 0.5 秒停頓就送出，字幕出現得快；auto 最長會等 2 秒
            "silence_duration": 0.5,
            # --- translation ---
            "target_language": "zh",
            # 獨立字幕視窗：同時顯示 2 句、每句最短顯示 0.3 秒，講得快也跟得上
            "subtitle_mode": {"sentences": 2, "min_display_ms": 300},
            "system_prompt": SYSTEM_PROMPT,
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

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    print(f"[write_settings] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
