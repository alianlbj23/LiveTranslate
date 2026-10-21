# LiveTranslate 一鍵安裝
# 第一次雙擊 LiveTranslate.bat 時（找不到 .venv）會自動執行；也可手動：powershell -File launcher\setup.ps1
# 會做的事：檢查驅動／網路／磁碟 → winget 安裝 Python、Git、Ollama（缺的才裝）→ 建 .venv 並安裝套件
#          → 預載 cuBLAS DLL → 依顯示卡大小選預設的辨識＋翻譯模型（hardware_profile.py）→ 下載 Whisper
#          （選到它才下載）→ 寫入預設設定。翻譯模型由啟動腳本在之後下載。
param(
    [switch]$Yes,          # 不問直接開始（測試用）
    [switch]$SkipModels    # 不下載 Whisper 模型、不寫設定（測試用）
)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Continue"   # 用 $LASTEXITCODE 逐步檢查，pip/winget 的 stderr 不會變成例外

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("setup_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
Start-Transcript -Path $LogFile -Append | Out-Null

. (Join-Path $PSScriptRoot "vram.ps1")   # Get-InstallChoice
. (Join-Path $PSScriptRoot "model.ps1")  # Set-LauncherModel

function Write-Step { param($m) Write-Host "`n[$((Get-Date).ToString('HH:mm:ss'))] $m" -ForegroundColor Cyan }
function Write-Ok   { param($m) Write-Host "  [完成] $m" -ForegroundColor Green }
function Write-Warn { param($m) Write-Host "  [注意] $m" -ForegroundColor Yellow }
function Write-Err  { param($m) Write-Host "  [錯誤] $m" -ForegroundColor Red }
function Fail {
    param($m)
    Write-Err $m
    Write-Host "  完整紀錄在：$LogFile" -ForegroundColor DarkGray
    Stop-Transcript | Out-Null
    if (-not $Yes) { Read-Host "按 Enter 關閉" | Out-Null }
    exit 1
}
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}
function Test-Cmd { param($name) return [bool](Get-Command $name -ErrorAction SilentlyContinue) }
function Find-Python {
    foreach ($v in "3.12", "3.11", "3.10") {
        try {
            $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path $exe.Trim())) { return $exe.Trim() }
        } catch {}
    }
    return $null
}
function Install-Winget {
    param($id, $label)
    Write-Step "安裝 $label（winget $id）..."
    & winget install --id $id -e --accept-source-agreements --accept-package-agreements --silent
    $code = $LASTEXITCODE
    # -1978335189 = 已安裝（APPINSTALLER_CLI_ERROR_PACKAGE_ALREADY_INSTALLED）
    if ($code -ne 0 -and $code -ne -1978335189) { Fail "$label 安裝失敗（winget 回傳 $code）" }
    Refresh-Path
    Write-Ok "$label 就緒"
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Magenta
Write-Host "   LiveTranslate 一鍵安裝" -ForegroundColor Magenta
Write-Host "==========================================" -ForegroundColor Magenta
Write-Host "  安裝位置：$Root"
Write-Host "  最多會下載約 12 GB（Python 套件約 5 GB；語音與翻譯模型依顯示卡大小選擇，最多約 8 GB），需要 20–40 分鐘。"
Write-Host "  過程中不用操作；若中途失敗，重新雙擊 LiveTranslate.bat 會從沒完成的地方繼續。"
if (-not $Yes) {
    $a = Read-Host "  按 Enter 開始，或輸入 n 取消"
    if ($a -match '^[nN]') { Stop-Transcript | Out-Null; exit 0 }
}

# ---------- 0. 事前檢查 ----------
Write-Step "檢查環境..."
if ($Root -match '[^\x00-\x7F]' -or $Root -match ' ') {
    Fail "安裝路徑不能有中文或空格：$Root`n  請把整個資料夾移到例如 D:\LiveTranslate 後再執行。"
}
$drive = Get-PSDrive -Name $Root.Substring(0, 1) -ErrorAction SilentlyContinue
if ($drive -and $drive.Free -lt 20GB) { Fail ("磁碟空間不足：{0} 槽只剩 {1:N1} GB，需要約 20 GB。" -f $drive.Name, ($drive.Free / 1GB)) }
if (-not (Test-Cmd winget)) {
    Start-Process "ms-windows-store://pdp/?ProductId=9NBLGGH4NNS1"
    Fail "找不到 winget。已幫你開啟 Microsoft Store 的「應用程式安裝程式」頁面，安裝後重新執行。"
}
if (-not (Test-Cmd nvidia-smi)) {
    Start-Process "https://www.nvidia.com/Download/index.aspx"
    Fail "找不到 NVIDIA 顯示卡驅動。已幫你開啟驅動下載頁，裝好並重新開機後再執行。"
}
$smi = (& nvidia-smi 2>$null) -join "`n"
$cudaVer = 0.0
if ($smi -match 'CUDA Version:\s*([0-9]+\.[0-9]+)') { $cudaVer = [double]$Matches[1] }
$cc = 0.0
try { $cc = [double]((& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1).Trim()) } catch {}
$gpuName = ((& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1) -as [string]).Trim()
$vramMB = 0
try { $vramMB = [int]((& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1).Trim()) } catch {}
Write-Ok "顯示卡：$gpuName（compute capability $cc，驅動支援 CUDA $cudaVer，VRAM $vramMB MB）"
if ($cc -ge 12.0) { $CudaIndex = "cu128" } else { $CudaIndex = "cu126" }
if ($cc -ge 12.0 -and $cudaVer -lt 12.8) { Fail "RTX 50 系列需要支援 CUDA 12.8 以上的驅動，目前是 $cudaVer，請更新 NVIDIA 驅動。" }
try {
    Invoke-WebRequest -Uri "https://pypi.org/simple/pip/" -Method Head -UseBasicParsing -TimeoutSec 10 | Out-Null
} catch {
    Fail "無法連上網路（pypi.org）：$($_.Exception.Message)"
}
Write-Ok "網路正常"

# ---------- 1. 前置軟體 ----------
Write-Step "檢查 Python / Git / Ollama..."
$PyExe = Find-Python
if (-not $PyExe) {
    Install-Winget "Python.Python.3.12" "Python 3.12"
    $PyExe = Find-Python
    if (-not $PyExe) { Fail "Python 已安裝但這個視窗還找不到它。請關閉視窗，重新雙擊 LiveTranslate.bat 一次。" }
}
Write-Ok "Python：$PyExe"
if (-not (Test-Cmd git)) { Install-Winget "Git.Git" "Git" } else { Write-Ok "Git 已安裝" }
$OllamaExe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
if (-not (Test-Path $OllamaExe)) {
    Install-Winget "Ollama.Ollama" "Ollama"
    if (-not (Test-Path $OllamaExe)) { Fail "Ollama 安裝後找不到 $OllamaExe" }
} else { Write-Ok "Ollama 已安裝" }

# ---------- 2. Python 虛擬環境與套件 ----------
Write-Step "建立 .venv 並安裝套件（PyTorch $CudaIndex，約 5 GB，最久的一步）..."
$VenvDir = Join-Path $Root ".venv"
$VenvPy  = Join-Path $VenvDir "Scripts\python.exe"
$Ready   = Join-Path $VenvDir ".livetranslate-ready"
if (-not (Test-Path $VenvPy)) {
    & $PyExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPy)) { Fail "建立 .venv 失敗" }
}
# pip 快取與暫存放在專案內，裝完刪掉（和上游 install.ps1 一樣，不塞爆系統碟）
$env:PIP_CACHE_DIR = Join-Path $Root ".pip-cache"
$env:TMP  = Join-Path $Root ".tmp"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null
Remove-Item -LiteralPath $Ready -Force -ErrorAction SilentlyContinue
& $VenvPy -m pip install --upgrade pip --quiet
& $VenvPy -m pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$CudaIndex"
if ($LASTEXITCODE -ne 0) { Fail "PyTorch 安裝失敗" }
& $VenvPy -m pip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { Fail "requirements.txt 安裝失敗" }
& $VenvPy -m pip install "yasbd-lib>=0.15,<1.0"
if ($LASTEXITCODE -ne 0) { Fail "yasbd-lib 安裝失敗" }
& $VenvPy -m pip check
if ($LASTEXITCODE -ne 0) { Fail "套件相依性檢查（pip check）失敗" }
Write-Ok "套件安裝完成"

# ---------- 3. 預載 cuBLAS DLL（CTranslate2 需要） ----------
Write-Step "設定 CUDA DLL 預載..."
$SitePackages = Join-Path $VenvDir "Lib\site-packages"
$DllModule = @'
"""Preload CUDA runtime DLLs shipped with torch so CTranslate2 (faster-whisper)
finds cublas64_12.dll even in processes that never import torch.
Loaded automatically via livetranslate_cuda_dlls.pth in this venv."""
import ctypes
import os
import sys


def _preload():
    lib = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib")
    lib = os.path.abspath(lib)
    if not os.path.isdir(lib):
        return
    try:
        os.add_dll_directory(lib)
    except (AttributeError, OSError):
        pass
    for name in ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll"):
        path = os.path.join(lib, name)
        if os.path.isfile(path):
            try:
                ctypes.WinDLL(path)
            except OSError:
                pass


if sys.platform == "win32":
    _preload()
'@
[IO.File]::WriteAllText((Join-Path $SitePackages "livetranslate_cuda_dlls.py"), $DllModule, (New-Object System.Text.UTF8Encoding $false))
[IO.File]::WriteAllText((Join-Path $SitePackages "livetranslate_cuda_dlls.pth"), "import livetranslate_cuda_dlls`n", (New-Object System.Text.ASCIIEncoding))
$cudaOk = (& $VenvPy -c "import torch; print(torch.cuda.is_available())" 2>$null | Select-Object -Last 1)
$ct2Ok  = (& $VenvPy -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())" 2>$null | Select-Object -Last 1)
if ("$cudaOk".Trim() -eq "True" -and [int]("$ct2Ok".Trim()) -ge 1) {
    Write-Ok "PyTorch 與 faster-whisper 都能使用 GPU"
} else {
    Write-Warn "GPU 檢查未通過（torch=$cudaOk, ctranslate2=$ct2Ok），程式仍可執行但會很慢。請確認驅動版本。"
}
Set-Content -LiteralPath $Ready -Value (Get-Date -Format o) -Encoding ascii
Remove-Item -Recurse -Force $env:PIP_CACHE_DIR, $env:TMP -ErrorAction SilentlyContinue

# ---------- 4. 語音模型與預設設定 ----------
$choice = $null
if (-not $SkipModels) {
    $SettingsFile = Join-Path $Root "user_settings.json"
    $fresh = -not (Test-Path $SettingsFile)
    # 不先下載用不到的模型：新安裝依顯示卡選；已有設定（重新安裝）就看設定裡用的是不是 Whisper turbo
    $needTurbo = $true
    if ($fresh) {
        Write-Step "依顯示卡（$vramMB MB）選擇預設的辨識與翻譯模型..."
        $choice = Get-InstallChoice -Python $VenvPy -Root $Root -TotalMB $vramMB
        if ($choice.level -eq "too_big") { Write-Warn $choice.message } else { Write-Ok $choice.message }
        $needTurbo = ($choice.asr_choice -eq "whisper-turbo")
    } else {
        $current = Invoke-HardwareAdvice -Python $VenvPy -Root $Root -Arguments @("launcher", $SettingsFile, "--total-mb", "$vramMB")
        if ($current) { $needTurbo = ($current.asr.key -eq "whisper:turbo") }  # 判斷不了就照舊下載
    }
    $WhisperPath = "medium"  # 沒下載 turbo 時的值：之後在控制面板改用 Whisper 會先問要不要下載
    if ($needTurbo) {
        Write-Step "下載語音辨識模型 faster-whisper large-v3-turbo（約 1.5 GB）..."
        $whisperOut = & $VenvPy -I (Join-Path $Root "launcher\download_whisper_turbo.py") $Root
        if ($LASTEXITCODE -ne 0) { Fail "Whisper 模型下載失敗" }
        $WhisperPath = ($whisperOut | Select-Object -Last 1).Trim()
        Write-Ok "模型位置：$WhisperPath"
    } elseif ($fresh) {
        Write-Ok "語音辨識使用 SenseVoice Small（約 0.9 GB），第一次開啟字幕程式時會自動下載。"
    } else {
        Write-Ok "目前的設定沒有用 Whisper turbo，不下載。"
    }

    if (-not $fresh) {
        Write-Ok "已有 user_settings.json，保留原設定"
    } else {
        Write-Step "寫入預設設定（日文 → 台灣繁體中文、即時預覽）..."
        & $VenvPy -I (Join-Path $Root "launcher\write_settings.py") $Root $WhisperPath "http://127.0.0.1:11434/v1" $choice.model $choice.asr_choice
        if ($LASTEXITCODE -ne 0) { Fail "寫入設定失敗" }
        # 啟動腳本依 selected_model.txt 下載並使用翻譯模型
        if (-not (Set-LauncherModel -Path (Join-Path $Root "launcher\selected_model.txt") -Model $choice.model)) {
            Fail "無法記下選擇的翻譯模型（launcher\selected_model.txt）"
        }
        Write-Ok "設定完成"
    }
}

$next = "接下來會啟動程式並下載翻譯模型（只需一次）"
if ($choice) { $next = "接下來會啟動程式並下載翻譯模型 {0}（約 {1:0.0} GB，只需一次）" -f $choice.label, [double]$choice.size_gb }
Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "   安裝完成！$next" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  紀錄：$LogFile" -ForegroundColor DarkGray
Stop-Transcript | Out-Null
exit 0
