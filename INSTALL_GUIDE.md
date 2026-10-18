# LiveTranslate 日文直播 → 中日雙字幕 安裝與使用指南

> 本文件依 2026-10-17 ～ 10-18 在這台電腦上**實際執行的結果**撰寫。所有元件都在 Windows 原生環境（PowerShell）執行，沒有使用 WSL 或 Docker。
> 專案已 fork 到 https://github.com/alianlbj23/LiveTranslate（`origin`），原作者為 `upstream`。本 fork 相對原版的修改：繁體中文介面、Whisper beam size 可設定（預設 greedy）、快速模式啟動腳本、本文件與 `launcher\` 工具。

---

## 1. 環境總覽

### 硬體 / 系統

| 項目 | 實際值 |
|---|---|
| 作業系統 | Windows 11 Pro 10.0.26200（系統地區設定為 ja-JP） |
| 顯示卡 | NVIDIA GeForce RTX 5070，12 GB VRAM（Blackwell，compute capability 12.0 / sm_120） |
| 顯示卡驅動 | 591.86（驅動支援 CUDA 13.1，可執行 cu128 版 PyTorch） |
| 記憶體 | 64 GB |
| 安裝路徑 | `D:\workspace\language_translate`（D 槽剩餘約 169 GB） |

### 軟體版本（全部實測）

| 元件 | 版本 / 位置 |
|---|---|
| git | 2.50.0.windows.1 |
| Python（venv 用） | 3.10.11（`%LOCALAPPDATA%\Programs\Python\Python310`） |
| 虛擬環境 | `D:\workspace\language_translate\.venv`（約 5.2 GB） |
| PyTorch / torchaudio | 2.11.0+cu128（CUDA 12.8，cuDNN 9.19） |
| CTranslate2 / faster-whisper | 4.8.3 / 1.2.1 |
| PyQt6 | 6.11.0 |
| openai（Python 套件） | 3.33.0 |
| 其他 | funasr 1.4.16、transformers 5.20.1、silero-vad 6.2.3、huggingface_hub 2.2.0、numpy 2.2.6、PyAudioWPatch 0.2.12.9 |
| LiveTranslate | fork 自 `TheDeathDragon/LiveTranslate` commit `190ef02`（2026-08-17），加上本 fork 的修改 |
| 語音辨識模型 | faster-whisper **large-v3-turbo**（`mobiuslabsgmbh/faster-whisper-large-v3-turbo`，約 1.5 GB，存於 `D:\workspace\language_translate\models\huggingface\hub\`） |
| Ollama（Windows 版） | 0.40.5，`%LOCALAPPDATA%\Programs\Ollama\ollama.exe` |
| 翻譯模型 | `qwen3.5:9b`（6.55 GB，存於 `%USERPROFILE%\.ollama\models`，執行時佔 5.6 GB VRAM，100% 在 GPU） |

### 這台電腦的特殊狀況（很重要）

- **WSL 裡面已經有另一個 Ollama 0.34.0**（Ubuntu-22.04 / 24.04 都在執行），它透過 `wslrelay.exe` 佔住了 `localhost:11434`，裡面只有 `qwen2.5:3b`。
- 依規則沒有動 WSL。Windows 版 Ollama 安裝後偵測到 11434 已有服務，就不會自己開服務。
- 因此桌面啟動腳本會**自動判斷**：11434 若是 Windows 版 Ollama 就直接用；若是 WSL 的且沒有 `qwen3.5:9b`，就把 Windows 版 Ollama 開在 **11435**，並自動把 LiveTranslate 的 API 位址改成對應的埠。你不需要手動處理。
- 副作用：在 PowerShell 打 `ollama --version` 會顯示 `0.34.0`（那是 WSL 那台回答的），Windows 版實際是 0.40.5。

---

## 2. 從零重裝的完整步驟

> 以下指令都在 **PowerShell** 執行。若只是想重新安裝，先看第 7 節解除安裝。

### 步驟 1：檢查環境

目的：確認顯示卡驅動、git、Python 版本。

```powershell
nvidia-smi
git --version
py -0p
```

預期：`nvidia-smi` 顯示 RTX 5070 且 `CUDA Version` ≥ 12.8；`py -0p` 列出 `-3.10-64`（LiveTranslate 只支援 3.10–3.12，3.13 不行）。
若沒有 Python 3.10：`winget install Python.Python.3.10`。

### 步驟 2：安裝 Windows 版 Ollama

目的：本機跑翻譯模型。

```powershell
winget install --id Ollama.Ollama --accept-source-agreements --accept-package-agreements --silent
```

預期：安裝到 `%LOCALAPPDATA%\Programs\Ollama`，右下角出現 Ollama 系統匣圖示，並在「啟動」資料夾加入 `Ollama.lnk`（開機自動啟動）。
安裝檔約 1 GB。新開一個 PowerShell 才會有 `ollama` 指令。

### 步驟 3：下載翻譯模型

目的：取得 `qwen3.5:9b`（約 6.5 GB）。

```powershell
# 若 11434 被 WSL 佔用，先把 Windows 版 Ollama 開在 11435（本次實際情況）
$env:OLLAMA_HOST = "127.0.0.1:11435"
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList serve -WindowStyle Hidden
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen3.5:9b
```

若 11434 沒有被佔用，省略 `$env:OLLAMA_HOST` 那行，直接 `ollama pull qwen3.5:9b` 即可。
預期：最後顯示 `success`，`ollama list` 看得到 `qwen3.5:9b`。

測試翻譯（關閉思考模式）：

```powershell
$body = '{"model":"qwen3.5:9b","stream":false,"think":false,"messages":[{"role":"user","content":"請把這句日文翻成台灣繁體中文，只輸出翻譯：今日はいい天気ですね"}]}'
Invoke-RestMethod -Uri http://127.0.0.1:11435/api/chat -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body)) | Select-Object -ExpandProperty message
```

實際結果：`content: 今天天氣真好啊。`，`thinking` 為空。第一次呼叫因為要把模型載入顯示卡，可能要等 1–2 分鐘；之後每句約 0.3–1 秒。

### 步驟 4：下載 LiveTranslate

```powershell
New-Item -ItemType Directory -Force D:\workspace | Out-Null
git clone https://github.com/TheDeathDragon/LiveTranslate D:\workspace\language_translate
```

它會安裝的東西（來自 `install.ps1` / `requirements.txt`）：Python venv、PyTorch + torchaudio（CUDA 版）、faster-whisper / CTranslate2、FunASR 與相依（SenseVoice 用）、transformers、silero-vad、PyQt6（字幕視窗）、openai（呼叫翻譯 API）、PyAudioWPatch（WASAPI 擷取系統聲音）、yasbd-lib（斷句）。

### 步驟 5：安裝相依套件（手動流程）

`install.bat` 會用 `Read-Host` 問你要 CUDA 還是 CPU，無法自動化，所以本次照 README 的手動流程做，等同於 install.ps1 的動作：

```powershell
Set-Location D:\workspace\language_translate
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 把 pip 快取放在專案內，裝完再刪（和官方腳本一樣）
$env:PIP_CACHE_DIR = "D:\workspace\language_translate\.pip-cache"; $env:TMP = "D:\workspace\language_translate\.tmp"; $env:TEMP = $env:TMP
New-Item -ItemType Directory -Force $env:TMP | Out-Null

# RTX 50 系列一定要 cu128（單一 wheel 約 2.75 GB）
.\.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install "yasbd-lib>=0.15,<1.0"
.\.venv\Scripts\python.exe -m pip check          # 預期：No broken requirements found.

Remove-Item -Recurse -Force .pip-cache, .tmp
Set-Content .venv\.livetranslate-ready (Get-Date -Format o) -Encoding ascii   # start.bat 會檢查這個標記
```

### 步驟 6：修正 CTranslate2 找不到 cuBLAS DLL

實測：在**沒有先 import torch** 的 Python 程序裡載入 faster-whisper 會出現
`RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`。
torch 的 wheel 其實已經附帶 `cublas64_12.dll`，只是別人找不到。解法是在 venv 加一個開機自動載入的小模組，把 torch 附帶的 DLL 先載進程序（不用再多下載 400 MB 的 `nvidia-cublas-cu12`）：

- `D:\workspace\language_translate\.venv\Lib\site-packages\livetranslate_cuda_dlls.py`
- `D:\workspace\language_translate\.venv\Lib\site-packages\livetranslate_cuda_dlls.pth`（內容只有一行 `import livetranslate_cuda_dlls`）

重裝時若這兩個檔案不見了，複製回去即可（內容見第 6 節）。
備用解法：`.\.venv\Scripts\pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`，然後把 `.venv\Lib\site-packages\nvidia\cublas\bin` 與 `nvidia\cudnn\bin` 加到 PATH。

### 步驟 7：驗證 GPU

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -c "import ctranslate2; print(ctranslate2.get_cuda_device_count(), ctranslate2.get_supported_compute_types('cuda'))"
```

實際輸出：`2.11.0+cu128 True NVIDIA GeForce RTX 5070`、`1 {'float16', 'int8_float16', ...}`。

### 步驟 8：下載 Whisper large-v3-turbo 並寫入設定

LiveTranslate 內建清單只有 tiny/base/small/medium/large-v3，但支援「本地 faster-whisper 模型資料夾」，所以用腳本把 turbo 下載到它的快取目錄：

```powershell
.\.venv\Scripts\python.exe -I launcher\download_whisper_turbo.py D:\workspace\language_translate
# 最後一行會印出模型資料夾路徑，貼到下一行的 <路徑>
.\.venv\Scripts\python.exe -I launcher\write_settings.py D:\workspace\language_translate "<路徑>" "http://127.0.0.1:11435/v1" "qwen3.5:9b"
```

`write_settings.py` 會產生 `user_settings.json`（第 3 節的所有設定都在裡面），之後 GUI 的修改也會存回這個檔。

### 步驟 9：建立桌面啟動與維護工具

已建立的檔案（重裝時若遺失，從本文件或備份復原）：

| 檔案 | 用途 |
|---|---|
| `D:\workspace\language_translate\日文雙字幕.bat` | **日常用這個**，點兩下即可 |
| `D:\workspace\language_translate\launcher\start_jp_subtitles.ps1` | 啟動邏輯：檢查/啟動 Ollama → 確認模型 → 校正 API 埠 → 啟動 LiveTranslate |
| `D:\workspace\language_translate\維護選單.bat` | 互動選單：切換 4b/9b 模型、更新、停止 Ollama、開本文件 |
| `D:\workspace\language_translate\檢查環境.bat` | 一鍵驗證 GPU / Whisper CUDA / Ollama / 翻譯 |
| `launcher\set_api_base.py`、`write_settings.py`、`download_whisper_turbo.py` | 上面腳本用的小工具 |

---

## 2a. 快速模式（本 fork 預設啟用）

目的：縮短「講完 → 字幕出現」的時間。每句的延遲由三段組成：等停頓（VAD）→ 語音辨識 → 翻譯，所以三段都調了：

| 項目 | 原版 | 本 fork | 效果 |
|---|---|---|---|
| Whisper `beam_size`（`config.yaml` → `asr.beam_size`） | 5（寫死） | **1**（greedy，可改回 5） | 語音辨識快約 20%，口語準確度略降（實測數字見下） |
| `interim_interval`（邊講邊出字幕的更新間隔） | 1.0 s | 1.0 s（維持） | 實測 GUI 下限就是 1.0 秒，且管線內建 1 秒冷卻，調低沒有效果 |
| `max_speech_duration`（長句強制切分） | 8.0 s | **5.0 s** | 長句不用等 8 秒 |
| 翻譯模型 `context_turns`（帶入的上下文句數） | 5 | **3** | 每句少處理約一半的提示詞 |
| Ollama `OLLAMA_KEEP_ALIVE=30m` | 閒置 5 分鐘卸載 | **閒置 30 分鐘才卸載** | 直播安靜一陣後第一句不用等 10–30 秒重載；程式正常結束時會立刻釋放（見下） |
| **關閉即釋放顯示卡** | 無 | **有** | 結束 LiveTranslate 時程式會呼叫 Ollama 卸載翻譯模型，啟動腳本接著關閉自己開的 Ollama 服務；Whisper 子程序隨程式結束釋放。直接關掉黑色視窗沒走正常流程時，模型最多 30 分鐘後自動卸載 |
| Ollama `OLLAMA_FLASH_ATTENTION=1` | 關 | **開** | 省顯示記憶體、提示詞處理較快 |
| 啟動時預先載入翻譯模型 | 無 | **有**（與載入 Whisper 同時進行） | 第一句就是正常速度 |
| VAD 靜音判定 `silence_mode` / `silence_duration` | auto（0.3–2.0 秒自適應） | **fixed 0.5 秒** | 講完停 0.5 秒就送出，不會等到 2 秒；代價是長句較容易被切成兩段 |
| `min_speech_duration` | 1.0 s | **0.6 s** | 短句（「はい」「やばい」）不用等併入下一句 |
| 字幕視窗最短顯示時間 `subtitle_mode.min_display_ms` | 1500 ms（寫死） | **300 ms**（改為可設定） | 原本每句至少佔 1.5 秒、新句還會把排隊中的前一句取消，講得快時字幕會晚 1.5 秒甚至漏句；已改成「排隊中的句子先顯示再接新句」，不再漏句 |
| 字幕視窗顯示句數 `subtitle_mode.sentences` | 1 | **2** | 上一句留著、新句直接接上，不用等 |
| **即時預覽（邊講邊翻）** `live_preview` | 無（只有整句講完才翻） | **開啟** | 每次增量辨識都把「還沒講完的那句」立刻顯示在字幕視窗，並先送去翻譯顯示暫時譯文；句子講完後由正式翻譯取代。這就是同步口譯式的「一聽到就出字幕」 |
| 增量辨識間隔 `interim_interval` | 1.0 s（GUI 下限 1.0、管線冷卻 1.0） | **0.5 s**（下限改為 0.3） | 預覽每 0.5 秒更新一次；辨識一次只要 0.15 秒，所以 0.5 秒綽綽有餘 |

實測（RTX 5070，faster-whisper large-v3-turbo，11.4 秒語音，取 3 次最快值）：

| compute_type | beam_size=5 | beam_size=1 |
|---|---|---|
| float16（採用） | 273 ms | **213 ms**（快約 20%） |
| int8_float16 | 302 ms | 236 ms（在 RTX 50 上反而比 float16 慢，不採用） |

辨識結果四種設定完全相同。注意：純辨識只要 0.2–0.3 秒，實際看直播時每句 0.8–1.9 秒的大部分時間花在等停頓（VAD）與第一次呼叫的暖機，所以 VAD 參數的調整比 beam size 更有感。

**字幕出現太慢時去哪裡調**（由影響大到小）：
0. 設定 → VAD / ASR → 時間參數 → 勾「增量識別」與「即時預覽（邊講邊翻）」，識別間隔 0.3–0.5 秒。這是最接近 Viitor 那種同步口譯效果的設定：講到一半字幕就會出現原文與暫時譯文，句尾再換成正式翻譯。暫時譯文偶爾會和正式翻譯不同，屬正常現象。
1. 設定 → VAD / ASR → 時間參數：「靜音模式」固定、「靜音時長」0.3–0.5 秒（越小越快、越容易切半句）、「最短語音」0.5 秒、「最長語音」4–5 秒。
2. 設定 → 字幕 → 顯示句數 2～3；`user_settings.json` 的 `subtitle_mode.min_display_ms`（毫秒，沒有 GUI 欄位）。
3. 翻譯模型編輯 → 上下文數 2～3（越少提示詞越短）。
4. 以上都調了還慢，看系統匣 → Show Log 裡 `ASR (xxx ms)` 與 `Translate (xxx ms)` 的數字：若 ASR 超過 1 秒，通常是顯示記憶體被其他程式吃掉（關 Wallpaper Engine / 遊戲）。

這些 Ollama 環境變數只在「由啟動腳本開的 Ollama」生效，所以啟動腳本會記錄自己開的 Ollama 的 PID（`launcher\ollama.pid`）；若發現 11434 上的是系統匣自動啟動的舊服務（沒有快速設定），會自動把它關掉並重新以快速設定啟動。這不會動到 WSL 裡的 Ollama。

想改回原版行為：`config.yaml` 的 `beam_size` 改 5；Settings → VAD / ASR 把時間參數調回；翻譯模型編輯對話框把上下文數改回 5。

### dev 分支的進一步優化（2026-10-18）

| 改善 | 做法 | 位置 |
|---|---|---|
| 字幕保證繁體、台灣用語 | 翻譯結果（含串流中的部分文字）先經 OpenCC `s2twp` 轉換再顯示，模型就算漏出簡體字也會被轉掉（例：软件→軟體、视频→影片、鼠标→滑鼠） | `config.yaml` → `translation.zh_tw_postprocess: true`；`translator.py` |
| Whisper 不再把上一句當提示 | `condition_on_previous_text: false`，避免「ご視聴ありがとうございました」之類的幻覺句一直重複 | `config.yaml` → `asr.whisper` |
| 幻覺句過濾 | 辨識結果若「只有」已知的假字幕句（日文／中文／英文共 21 句，可自行增減）就直接丟掉，不送翻譯 | `asr.whisper.hallucination_phrases` |
| 可加入人名／術語 | `asr.whisper.hotwords` 填 VTuber 名字、遊戲術語（逗號分隔），Whisper 會偏向辨識成這些詞 | `config.yaml` |
| 可加入風格提示 | `asr.whisper.initial_prompt`，例如「こんにちは。今日は、配信を始めます。」可讓輸出帶標點，斷句更準（用 Anime-Whisper 時請留空） | `config.yaml` |
| 翻譯提示詞加範例 | 系統提示詞多了三組「日文 → 台灣口語」範例，用語更一致；提示詞會被 Ollama 快取，不增加延遲 | `user_settings.json` → `system_prompt` |
| Ollama KV cache 8-bit | `OLLAMA_KV_CACHE_TYPE=q8_0`（需搭配 flash attention），顯示記憶體再省一些，品質幾乎無差 | `launcher\start_jp_subtitles.ps1` |

**翻譯模型比較**（同一台機器、8 句直播口語、暖機後逐句測，/v1 端點）：

| 模型 | 平均延遲 | 顯示記憶體 | 品質（我逐句看過） | 結論 |
|---|---|---|---|---|
| `qwen3.5:9b`（Q4_K_M，關閉思考，帶 3 句上下文） | **250 ms／句** | 5.6 GB | 口語自然、理解直播用語（超留、聯動、BOSS 打不過）、台灣用語 | **維持預設** |
| Tencent HY-MT1.5-7B（Q4_K_M，專用翻譯模型，需自建模板） | 201 ms／句 | 4.6 GB | 比 1.8B 通順，但仍不懂直播用語（スパチャ→「斯帕查」、やばい→「太嚇人」），沒有上下文 | 不採用 |
| Tencent HY-MT1.5-1.8B（Q8_0，專用翻譯模型，需自建模板） | 140 ms／句 | 1.9 GB | 偏書面、不懂網路用語（スパチャ→「Spa 服務」、やばい→「太可怕」），偶爾漏出控制符 | 顯示記憶體真的不夠時的備案 |

HY-MT 若要使用：直接 `ollama pull hf.co/tencent/HY-MT1.5-1.8B-GGUF:Q8_0` 的版本對話模板不正確（只會輸出「onse」），必須照官方 README 用 Modelfile 建立（`TEMPLATE` 含 `<|startoftext|>`、`<|extra_4|>`、`<|extra_0|>`、`<|eos|>`），本機已建好名為 `hy-mt1.5-1.8b` 與 `hy-mt1.5-7b` 的版本（不用可以 `ollama rm` 刪掉，共 6.5 GB）。它沒有 system 角色，提示詞要用「将以下文本翻译为繁體中文，注意只需要输出翻译后的结果，不要额外解释：」格式，所以在 LiveTranslate 需勾「無 system 角色」並把系統提示詞換成這一句。


查過但沒採用的技術：
- **Qwen3-ASR-1.7B**（2026-01）：日文 CER 比 Whisper turbo 略好（0.140 vs 0.184），但速度約慢 3 倍，而且要另外整合 transformers 推論；直播字幕以延遲優先，維持 Whisper large-v3-turbo。
- **kotoba-whisper**（日文蒸餾版 Whisper）：速度快，但沒有可靠的 v2.2 評測與 CTranslate2 版本資訊，先不換。
- **投機解碼（speculative decoding）**：Ollama 目前不支援；每句翻譯只產生十幾個 token，收益也有限。
- **int8_float16**：實測在 RTX 5070 上比 float16 慢，不用。

### 繁體中文介面

本 fork 新增 `i18n/zh_TW.yaml`（台灣用語）與 `i18n/CHANGELOG_zh_TW.md`，`user_settings.json` 的 `ui_lang` 已設為 `zh_TW`。Settings → VAD / ASR → 介面語言 可在 English / 简体中文 / 繁體中文 之間切換（需重新啟動）。字幕字型改為 Microsoft JhengHei，才有台灣慣用字形。

---

## 3. LiveTranslate GUI 設定清單

這些值**已經由 `user_settings.json` 預填**，啟動後不需要再改；這裡列出是讓你知道在 GUI 哪裡改。介面語言已設為繁體中文（欄位名稱依 `i18n/zh_TW.yaml`）。

打開設定：字幕浮動視窗右上角 **Settings** 按鈕，或系統匣圖示右鍵。

### 分頁「VAD / ASR」

| 欄位（實際名稱） | 要選 / 填的值 | 備註 |
|---|---|---|
| 引擎: | **Whisper** | 備選：**Anime-Whisper**（VTuber / 動畫腔適用，第一次選會自動下載約 1.5 GB） |
| 語言提示: | **ja - 日本語** | 固定日文，避免誤判成中文或韓文 |
| 裝置: | **cuda:0 (NVIDIA GeForce RTX 5070)** | |
| 音訊: | 第二個選項（系統預設輸出裝置） | 這就是 WASAPI loopback，會擷取**所有**系統聲音，見第 6 節「只擷取 Chrome」 |
| 麥克風: | 第一個選項（不混入） | |
| 下載來源: | **HuggingFace** | |
| Whisper 模型（群組）: | **本地: mobiuslabsgmbh/faster-whisper-large-v3-turbo** | 狀態會顯示「本地模型」 |
| Whisper 補零: | 0.5 | 預設 |
| VAD 模式 / Silero 閾值 | silero / 0.3 | 預設 |
| 時間參數 | 最短 1.0 s、最長 **5.0 s**、靜音自動、增量間隔 1.0 s | 快速模式 |

### 分頁「翻譯」

| 欄位（實際名稱） | 要選 / 填的值 |
|---|---|
| 目前模型 (用於翻譯管線) | **Ollama qwen3.5:9b** |
| 模型設定 → 編輯該模型 → API 位址: | `http://127.0.0.1:11435/v1`（啟動腳本會視情況自動改成 11434） |
| API 金鑰: | `ollama` |
| 模型: | `qwen3.5:9b` |
| 代理: | **不使用代理** |
| 上下文數: | **3** |
| 禁用思考: | **reasoning_effort=none - OpenAI / Grok**（實測 Ollama 的 /v1 只認這個，其他選項模型照樣思考、翻譯會變空白） |
| 串流輸出 (逐字輸出) | ✅ 勾選 |
| 結構化輸出 (JSON 模式) | ❌ |
| 無 system 角色 | ❌ |
| extra_body (JSON) | 留空 |
| 系統提示詞 → 預設: | **自定义**（已填入「一律使用台灣繁體中文…」的提示詞） |

### 字幕浮動視窗（主畫面）

| 位置 | 值 |
|---|---|
| 第二列「目標」下拉選單 | **zh**（程式沒有 zh-TW 選項，繁體由系統提示詞保證） |
| 原文 + 翻譯同時顯示 | 預設即如此：每句上面是日文原文（灰色小字），下面是中文翻譯 |
| Click-through / Top-most | 看直播時建議都勾選，字幕才會浮在瀏覽器之上且不擋滑鼠 |

### 分頁「樣式」/「字幕」
純外觀（字型、顏色、預設配色），依喜好調整。本 fork 預設字型已是 Microsoft JhengHei（台灣慣用字形）。

---

## 4. 日常使用方式

1. 開啟 Chrome 的 YouTube 直播。
2. 點兩下 `D:\workspace\language_translate` 裡的 **`日文雙字幕.bat`**（可自行建立桌面捷徑）。黑色視窗會依序顯示 Ollama 檢查、模型檢查，然後出現字幕浮動視窗；語音模型載入約 5–10 秒。
3. 把字幕視窗拖到想要的位置，勾 **Top-most** 與 **Click-through**。
4. 直播有人講話後約 1–3 秒會出現日文原文與中文翻譯。
5. **關閉**：字幕視窗右上角 **結束**，或右下角系統匣的 LiveTranslate 圖示 → 結束。關閉黑色視窗也會一起結束。
   結束時程式會自動卸載翻譯模型並關閉啟動腳本開的 Ollama，顯示記憶體會全部釋放（黑色視窗會顯示「已卸載」「Ollama 服務已關閉」）。若是直接關掉黑色視窗，模型最多 30 分鐘後自動卸載；想立刻釋放用 `維護選單.bat` → 6。
6. **切換語音辨識模型**：Settings → VAD / ASR → 引擎 改成 Anime-Whisper（或 Whisper 模型下拉改 large-v3 / medium）。切換會自動重載，不用重啟。
7. **切換翻譯模型**：`D:\workspace\language_translate\維護選單.bat` → 3（qwen3.5:4b，顯示記憶體不足時）或 4（qwen3.5:9b）。
8. **更新 LiveTranslate**：`維護選單.bat` → 5（執行上游的 `update.bat`，不會動到 PyTorch）。
9. 每次執行會在 `D:\workspace\language_translate\transcripts\` 留下原文 / 翻譯逐字稿（可在 Settings → 逐字稿儲存 關閉）。
10. **兩種字幕視窗的操作方式**
    - 主浮動視窗（有「LiveTranslate」標題列的那個）：拖標題列移動，右下角有縮放把手；勾「滑鼠穿透」後只有標題列能點。
    - 獨立字幕視窗（主視窗按「字幕」或系統匣「字幕視窗」開啟，純文字、適合疊在影片上）：**左鍵按住任何位置拖曳**移動；**拖右邊緣或右下角**改寬度，文字會自動換行、高度自動配合；滑鼠移上去會顯示虛線框與右下角把手。位置與寬度調好後**雙擊**即鎖定（開啟滑鼠穿透，滑鼠經過不會再影響它）。要再調整：右下角系統匣 LiveTranslate 圖示 → 取消勾選「字幕滑鼠穿透」。寬度也可在 設定 → 字幕 → 視窗寬度 精確輸入。

---

## 5. 驗證方式

最簡單：點兩下 `D:\workspace\language_translate\檢查環境.bat`，應看到：

1. `nvidia-smi` 列出 RTX 5070。
2. `torch 2.11.0+cu128 | cuda available: True | NVIDIA GeForce RTX 5070`
3. `ctranslate2 4.8.3 | faster-whisper 1.2.1 | cuda devices: 1`
4. 連接埠 11434 / 11435 各是誰在回應、有哪些模型（程序名 `ollama` 是 Windows 版，`wslrelay` 是 WSL 的）。
5. 翻譯測試輸出「今天天氣真好…」，thinking 長度 0。

字幕是否正常：啟動後播放任何日文影片，字幕視窗應在幾秒內出現日文與中文；系統匣 → Show Log 可看到 `ASR [ja] (xxx ms): …` 與 `Translate (xxx ms): …` 兩種訊息。本次實測 ASR 約 0.8–1.9 秒、翻譯約 0.6–1.2 秒。

---

## 6. 疑難排解

### 本次實際遇到的問題與解法

| 現象 | 原因 | 解法 |
|---|---|---|
| Windows 版 Ollama 裝好但 `localhost:11434` 回應的是 0.34.0，`ollama pull` 把模型拉進 WSL | WSL 內的 Ollama 透過 wslrelay 佔住 11434 | 啟動腳本自動改用 11435；手動時設 `$env:OLLAMA_HOST="127.0.0.1:11435"` 再執行 `ollama serve` / `ollama pull` |
| `RuntimeError: Library cublas64_12.dll is not found` | CTranslate2 wheel 不含 cuBLAS，只有 torch 載入後才找得到 | venv 內的 `livetranslate_cuda_dlls.pth` 會預先載入 torch 附帶的 DLL（已安裝）。模組內容：用 `ctypes.WinDLL` 載入 `torch\lib\cudart64_12.dll`、`cublasLt64_12.dll`、`cublas64_12.dll` |
| 翻譯一直空白，Log 出現「Empty translation but 256 completion tokens were used」 | Ollama 的 OpenAI 相容端點忽略 `think:false`、`enable_thinking`、`chat_template_kwargs` 等，qwen3.5 把 token 全拿去思考 | 模型設定的「禁用思考」改成 **reasoning_effort=none**（實測唯一有效） |
| `install.bat` 跑到一半停住等輸入 | 它用 Read-Host 問 CUDA/CPU | 照第 2 節步驟 5 手動安裝 |
| 黑色視窗中文變亂碼 | 系統地區是日文，主控台預設 cp932 | `.bat` 內已先 `chcp 65001`；PowerShell 腳本存成 UTF-8 with BOM |
| Log 出現 `ASR worker RSS grew … over baseline; recycling` 然後重載模型約 4 秒 | 程式的記憶體保護機制，第一次推論後 worker 記憶體增長超過門檻 | 正常現象，重載後繼續運作；若太頻繁可在 VAD / ASR 把模型換成 medium |

### 常見問題

- **顯示記憶體不足 / 翻譯變很慢**：實測在桌面開了很多程式（Chrome、Discord、Steam、Wallpaper Engine 等已吃 3.7 GB）時，Whisper turbo + qwen3.5:9b 會用到 11.6 / 12.2 GB，非常接近上限。先關掉 Wallpaper Engine、多餘的 Chrome 分頁；仍不夠就用 `維護選單.bat` → 3 換成 `qwen3.5:4b`（約 3 GB VRAM）。
- **玩遊戲時不能同時使用**：遊戲本身會吃掉大部分 VRAM，Ollama 會被迫改用 CPU/RAM，翻譯延遲變成 10 秒以上甚至失敗。玩遊戲前先結束 LiveTranslate（用「結束」按鈕或系統匣 → 結束，顯示記憶體會自動釋放）；若只是關掉黑色視窗，再用 `維護選單.bat` → 6 立刻釋放。
- **背景音樂太大聲、辨識出奇怪的句子**（實測在沒人講話時辨識出「ご視聴ありがとうございました」）：這是 Whisper 對音樂/靜音的幻覺。調高 Settings → VAD / ASR → Silero VAD 閾值 到 0.5–0.6；或把引擎換成 Anime-Whisper（對日文配音較穩）；YouTube 端調低背景音樂無法做到，只能靠 VAD。
- **字幕出現簡體中文**：系統提示詞已強制台灣繁體；若仍出現，確認 翻譯 分頁的「預設」是 **自訂** 而不是「日常聊天」等預設（那些預設是英文提示詞，只說 Chinese）。也可在提示詞再加一句「輸出前檢查沒有任何簡體字」。
- **只想擷取 Chrome 的聲音**：LiveTranslate 只能擷取「某個輸出裝置」的全部聲音，沒有按程式擷取。替代做法：Windows 設定 → 系統 → 音效 → 音量混音器 → 把 Chrome 的「輸出裝置」改成另一個裝置（例如螢幕 HDMI 喇叭或虛擬音效裝置），然後在 LiveTranslate 的「音频」選那個裝置；其他程式維持原本輸出就不會被收進去。最簡單的方式則是看直播時把 Spotify / Discord 等靜音。
- **字幕視窗不見了**：系統匣圖示 → Show，或 Settings 內「重設視窗位置」。
- **開機後第一次啟動很久**：Ollama 第一次把 6.5 GB 模型載入顯示卡需 30–120 秒，之後就快了。
- **重開機後 API 埠變了**：正常。Windows 版 Ollama 開機自動啟動，若比 WSL 早搶到 11434，啟動腳本就會把設定改回 11434；兩種情況都會自動處理。
- **`ollama` 指令找不到**：安裝程式已加到使用者 PATH，開新的 PowerShell 即可；或用完整路徑 `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`。

---

## 7. 解除安裝

1. 關閉 LiveTranslate 與 Ollama（`維護選單.bat` → 6，或工作管理員結束 `ollama.exe`、`ollama app.exe`）。
2. 刪除整個資料夾 `D:\workspace\language_translate`（含 venv、Whisper 模型、逐字稿，約 7 GB）。
3. （啟動用的 .bat 都在專案資料夾內，會一起刪除）。
4. 移除 Ollama：

   ```powershell
   winget uninstall --id Ollama.Ollama
   Remove-Item -Recurse -Force "$env:USERPROFILE\.ollama"      # 模型檔，約 6.1 GB
   ```

   解除安裝程式會一併移除「啟動」資料夾的 `Ollama.lnk` 與使用者 PATH 項目。
5. 以上不會影響 WSL 內的 Ollama、系統 Python 3.10 / 3.13、或其他 Python 專案（本次安裝只在 `.venv` 內裝套件，沒有改任何系統環境變數）。
