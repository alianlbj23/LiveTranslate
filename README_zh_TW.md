# LiveTranslate

[English](README.md) | [简体中文](README_zh.md) | **繁體中文**

Windows 即時音訊翻譯工具。捕獲系統音訊（WASAPI loopback）和可選的麥克風輸入，語音識別後呼叫 LLM API 翻譯，結果顯示在透明浮動字幕視窗上。

適用於看外語影片、直播、語音對話等場景——無需修改播放器，全域性音訊捕獲即開即用。

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Windows](https://img.shields.io/badge/Platform-Windows-0078d4)
![License](https://img.shields.io/badge/License-MIT-green)

> ### 關於這個 fork（alianlbj23/LiveTranslate）
> 在原專案基礎上加入：**繁體中文（台灣）介面**（`i18n/zh_TW.yaml`）、Whisper `beam_size` 可在 `config.yaml` 設定（預設 greedy）、
> 專為「YouTube 日文直播 → 日文 + 台灣繁體中文雙字幕、本機 Ollama 翻譯」設計的單一入口腳本（`LiveTranslate.bat`：啟動／檢查環境／切換模型／更新／停止 Ollama）與快速模式。
> 完整的安裝、設定、疑難排解請看 [INSTALL_GUIDE.md](INSTALL_GUIDE.md)。

## 截圖

![LiveTranslate](screenshot/zh.png)

## 安裝影片

[![安裝演示](https://img.shields.io/badge/Bilibili-安裝演示-00A1D6?logo=bilibili)](https://www.bilibili.com/video/BV1K2Awz6Euw) 適用於看外語影片、直播、ASMR等場景，也可以語音輸入即時並行翻譯多種語音

## 功能特性

- **即時翻譯管線**：系統音訊 → VAD → ASR → LLM 翻譯 → 字幕顯示
- **多 ASR 引擎**：faster-whisper、SenseVoice、FunASR Nano、Anime-Whisper
- **遠端 ASR**：透過 HTTP 把語音識別放到 GPU 機器上跑 —— 見 [REMOTE_ASR.md](REMOTE_ASR.md)
- **相容任意 OpenAI 格式 API**：DeepSeek、Grok、Qwen、GPT、Ollama、vLLM 等
- **串流翻譯顯示**：翻譯結果逐字即時顯示
- **模型獨立設定**：串流傳輸、結構化輸出(JSON)、上下文歷史、禁用思考
- **麥克風混音**：可選將麥克風輸入混合到系統音訊一起識別
- **低延遲 VAD**：32ms 音訊塊 + Silero VAD，自適應靜音檢測
- **透明浮動窗**：永遠置頂、滑鼠穿透、可拖拽，14 種配色主題
- **CUDA 加速**：ASR 模型 GPU 推理
- **模型自動管理**：首次啟動向導，支援 ModelScope / HuggingFace 雙源
- **內建效能測試**：對比翻譯模型速度和質量

## 更新紀錄

檢視 [中文更新紀錄](i18n/CHANGELOG_zh.md) | [English Changelog](i18n/CHANGELOG_en.md)

## 系統要求

- **作業系統**：Windows 10/11
- **Python**：3.10–3.12（免安裝版免裝）
- **GPU**（推薦）：NVIDIA 顯示卡 + CUDA 12.6（RTX 50 系列等 Blackwell 架構需要 CUDA 12.8）
- **網路**：需要訪問翻譯 API

## 快速開始

### 免安裝版（免裝 Python，推薦新手）

從 [Releases](https://github.com/TheDeathDragon/LiveTranslate/releases) 下載 `LiveTranslate-portable-*.zip`，解壓後雙擊 **`start.bat`** 即可。首次執行會自動下載便攜版 Python 3.12 並按顯示卡安裝依賴，無需預裝任何 Python。

### 從原始碼安裝

```bash
git clone https://github.com/TheDeathDragon/LiveTranslate.git
cd LiveTranslate
```

雙擊 **`install.bat`** 一鍵安裝——指令碼會自動：
1. 檢測 Python 3.10–3.12（未安裝則透過 winget 自動安裝）
2. 建立虛擬環境
3. 檢測 NVIDIA 顯示卡，選擇 CUDA / CPU 版 PyTorch
4. 安裝全部依賴

安裝完成後雙擊 **`start.bat`** 啟動。

更新時雙擊 **`update.bat`**——自動拉取最新程式碼並更新依賴（未安裝 Git 會透過 winget 自動安裝）。

<details>
<summary>手動安裝</summary>

```bash
python -m venv .venv
.venv\Scripts\activate

# PyTorch（三選一）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126  # CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128  # CUDA（RTX 50 系列）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu    # 僅 CPU

# 依賴
pip install -r requirements.txt

# 啟動
.venv\Scripts\python.exe main.py
```

</details>

## 首次使用

1. 彈出設定嚮導——選擇下載來源（ModelScope 適合中國大陸，HuggingFace 適合台灣與海外）和快取路徑
2. 自動下載 Silero VAD + SenseVoice 模型（約 1GB）
3. 下載完成後進入主介面

## 設定翻譯 API

設定 → 翻譯分頁：

| 引數 | 示例 |
|------|------|
| API Base | `https://api.deepseek.com/v1` |
| API Key | 你的金鑰 |
| Model | `deepseek-chat` |
| 代理 | `none` / `system` / 自定義位址 |

## 架構

```
Audio (WASAPI 32ms) → VAD (Silero) → ASR → LLM Translation → Overlay
         ↑ 可選麥克風混音
```

```
main.py                 主入口，管線編排
├── audio_capture.py    WASAPI loopback + 麥克風混音
├── vad_processor.py    Silero VAD
├── asr_engine.py       faster-whisper 後端
├── asr_funasr.py       統一 FunASR 模型選擇後端
├── asr_sensevoice.py   SenseVoice 後端
├── asr_funasr_nano.py  FunASR Nano 後端
├── asr_anime_whisper.py Anime-Whisper 後端 (日語動畫/Galgame)
├── asr_remote.py        遠端 Whisper 客戶端 (→ asr_server.py, 見 REMOTE_ASR.md)
├── translator.py       OpenAI 相容翻譯客戶端 (串流/JSON/上下文)
├── model_manager.py    模型下載與快取管理
├── subtitle_overlay.py PyQt6 透明浮動窗
├── control_panel.py    設定面板 UI (7 個分頁)
├── dialogs.py          設定嚮導、下載、模型設定對話方塊
└── benchmark.py        翻譯效能測試
```

## 致謝

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 基於 CTranslate2 的 Whisper 推理
- [FunASR](https://github.com/modelscope/FunASR) — SenseVoice / Fun-ASR-Nano
- [Anime-Whisper](https://huggingface.co/litagin/anime-whisper) — 日語動畫/Galgame 專用 ASR
- [Silero VAD](https://github.com/snakers4/silero-vad) — 語音活動檢測

## Star History

<a href="https://www.star-history.com/?repos=TheDeathDragon%2FLiveTranslate&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=TheDeathDragon/LiveTranslate&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=TheDeathDragon/LiveTranslate&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=TheDeathDragon/LiveTranslate&type=date&legend=top-left" />
 </picture>
</a>

## 許可證

[MIT License](LICENSE)
