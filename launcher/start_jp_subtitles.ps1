# LiveTranslate 日文雙字幕 一鍵啟動（快速模式）
# 1) 確認 Windows 版 Ollama 以「快速設定」在跑（模型常駐不卸載 + flash attention），沒有就啟動
# 2) 確認翻譯模型存在，並在背景先把模型載入顯示卡
# 3) 把 LiveTranslate 的 API 位址對準正確的 Ollama 連接埠
# 4) 啟動 LiveTranslate
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Continue"

$Root     = Split-Path -Parent $PSScriptRoot   # launcher\ 的上一層，搬資料夾也不用改
$Python   = Join-Path $Root ".venv\Scripts\python.exe"
$Ollama   = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
$ModelFile = Join-Path $Root "launcher\selected_model.txt"   # 選單切換的模型，git 不追蹤
. (Join-Path $Root "launcher\model.ps1")
$Model    = Get-LauncherModel -Path $ModelFile
$Settings = Join-Path $Root "user_settings.json"
$Helper   = Join-Path $Root "launcher\set_api_base.py"
$UpdateScript = Join-Path $Root "launcher\update.ps1"
$WatchdogScript = Join-Path $Root "launcher\gpu_watchdog.ps1"
$PidFile  = Join-Path $Root "launcher\ollama.pid"

function Write-Info { param($m) Write-Host "[資訊] $m" -ForegroundColor Cyan }
function Write-Ok   { param($m) Write-Host "[完成] $m" -ForegroundColor Green }
function Write-Warn { param($m) Write-Host "[注意] $m" -ForegroundColor Yellow }
function Write-Err  { param($m) Write-Host "[錯誤] $m" -ForegroundColor Red }

function Test-OllamaAlive {
    param([int]$Port)
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/version" -TimeoutSec 2
        return [string]$r.version
    } catch { return $null }
}

function Get-PortOwner {
    # 回傳 @{ Name; Id }
    param([int]$Port)
    try {
        $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1
        if ($c) { $p = Get-Process -Id $c.OwningProcess -ErrorAction Stop; return @{ Name = $p.Name; Id = $p.Id } }
    } catch {}
    return @{ Name = "unknown"; Id = 0 }
}

function Test-HasModel {
    param([int]$Port)
    try {
        $tags = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/tags" -TimeoutSec 3
        return (@($tags.models | ForEach-Object { $_.name }) -contains $Model)
    } catch { return $false }
}

function Test-OurServer {
    # 只有由本腳本啟動的 Ollama 才帶有快速設定（常駐 + flash attention）
    param([int]$OwnerPid)
    if ($OwnerPid -le 0 -or -not (Test-Path $PidFile)) { return $false }
    try { return ([int](Get-Content $PidFile -ErrorAction Stop) -eq $OwnerPid) } catch { return $false }
}

function Stop-WindowsOllama {
    # 關掉系統匣 app 與它帶起來的 serve（不影響 WSL 裡的 Ollama）
    Get-Process -Name "ollama app", "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

function Start-WindowsOllama {
    param([int]$Port)
    if (-not (Test-Path $Ollama)) {
        Write-Err "找不到 Windows 版 Ollama：$Ollama"
        Write-Err "請執行 winget install Ollama.Ollama 後再試。"
        return $false
    }
    Write-Info "啟動 Windows 版 Ollama（連接埠 $Port，模型常駐、flash attention）..."
    $env:OLLAMA_HOST            = "127.0.0.1:$Port"
    $env:OLLAMA_KEEP_ALIVE      = "30m"     # 閒置 30 分鐘才卸載（正常結束時會立刻釋放，這是保險）
    $env:OLLAMA_FLASH_ATTENTION = "1"       # 省顯示記憶體、提示詞處理較快
    $env:OLLAMA_KV_CACHE_TYPE   = "q8_0"    # KV cache 量化成 8-bit：顯示記憶體省一半，翻譯品質幾乎無差
    $env:OLLAMA_NUM_PARALLEL    = "2"       # 2 個槽：正式翻譯與即時預覽不互相排隊；KV cache 仍比預設 4 槽省
    $env:OLLAMA_CONTEXT_LENGTH  = "2048"    # 字幕翻譯一次只需幾百個 token，2048 夠用且省顯示記憶體
    $p = Start-Process -FilePath $Ollama -ArgumentList "serve" -WindowStyle Hidden -PassThru
    Set-Content -Path $PidFile -Value $p.Id -Encoding ascii
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if (Test-OllamaAlive $Port) { return $true }
    }
    Write-Err "Ollama 在 30 秒內沒有回應。"
    return $false
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Magenta
Write-Host "   LiveTranslate 日文 -> 台灣繁體中文 雙字幕" -ForegroundColor Magenta
Write-Host "==========================================" -ForegroundColor Magenta
Write-Host ""

if (-not (Test-Path $Python)) {
    Write-Err "找不到 $Python，請先完成安裝（參考 INSTALL_GUIDE.md）。"
    Read-Host "按 Enter 關閉"
    exit 1
}

# ---- Step 0: 自動更新（離線或本機有修改時略過，永不擋啟動） ----
if (Test-Path $UpdateScript) { & $UpdateScript }

# ---- Step 0b: 顯示記憶體檢查 ----
. (Join-Path $Root "launcher\vram.ps1")
$freeMB = Get-FreeVramMB
if ($freeMB -ge 0) { Write-Info ("可用顯示記憶體 {0:N1} GB" -f ($freeMB / 1024)) }
Test-LowVram -FreeMB $freeMB | Out-Null

# ---- Step 1: Ollama ----
$Port = 11434
$ver = Test-OllamaAlive $Port
if ($ver) {
    $owner = Get-PortOwner $Port
    if ($owner.Name -eq "ollama") {
        if (Test-OurServer $owner.Id) {
            Write-Ok "Windows 版 Ollama $ver 已以快速設定在連接埠 $Port 執行。"
        } else {
            Write-Warn "連接埠 $Port 的 Windows 版 Ollama 是系統匣自動啟動的（沒有快速設定），重新以快速設定啟動..."
            Stop-WindowsOllama
            if (-not (Start-WindowsOllama $Port)) { Read-Host "按 Enter 關閉"; exit 1 }
            Write-Ok "Windows 版 Ollama 已重新啟動（連接埠 $Port）。"
        }
    } elseif (Test-HasModel $Port) {
        Write-Warn "連接埠 $Port 的 Ollama 不是 Windows 版（程序：$($owner.Name)，常見原因是 WSL 內的 Ollama），但它有 $Model，直接使用。"
    } else {
        Write-Warn "連接埠 $Port 被 $($owner.Name) 佔用（通常是 WSL 內的 Ollama），而且裡面沒有 $Model。"
        $Port = 11435
        $alt = Test-OllamaAlive $Port
        if ($alt) {
            $owner2 = Get-PortOwner $Port
            if (Test-OurServer $owner2.Id) {
                Write-Ok "Windows 版 Ollama 已以快速設定在備用連接埠 $Port 執行。"
            } else {
                Write-Warn "備用連接埠 $Port 的 Ollama 沒有快速設定，重新啟動..."
                Stop-WindowsOllama
                if (-not (Start-WindowsOllama $Port)) { Read-Host "按 Enter 關閉"; exit 1 }
            }
        } elseif (-not (Start-WindowsOllama $Port)) {
            Read-Host "按 Enter 關閉"; exit 1
        } else {
            Write-Ok "Windows 版 Ollama 已在備用連接埠 $Port 啟動。"
        }
    }
} else {
    if (-not (Start-WindowsOllama $Port)) { Read-Host "按 Enter 關閉"; exit 1 }
    Write-Ok "Windows 版 Ollama 已啟動（連接埠 $Port）。"
}

# ---- Step 2: model ----
if (Test-HasModel $Port) {
    Write-Ok "翻譯模型 $Model 已就緒。"
} else {
    Write-Warn "連接埠 $Port 的 Ollama 沒有 $Model，開始下載（約 6 GB，只需一次）..."
    $env:OLLAMA_HOST = "127.0.0.1:$Port"
    & $Ollama pull $Model
    if (-not (Test-HasModel $Port)) {
        Write-Err "模型下載失敗，請檢查網路後再試。"
        Read-Host "按 Enter 關閉"; exit 1
    }
}

# ---- Step 3: point LiveTranslate at the right port ----
if (Test-Path $Settings) {
    & $Python -I $Helper $Settings "http://127.0.0.1:$Port/v1" $Model
}

# ---- Step 3b: 看門狗（隱藏視窗，叉掉本視窗或程式當掉時也會釋放顯示卡） ----
if (Test-Path $WatchdogScript) {
    $wdLog = Join-Path $Root "logs\gpu_watchdog.log"
    New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null
    Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$WatchdogScript`"",
        "-ParentPid", $PID, "-Port", $Port, "-PidFile", "`"$PidFile`"", "-LogFile", "`"$wdLog`""
    ) | Out-Null
}

# ---- Step 4: LiveTranslate（先啟動程式，趁它載入 Whisper 的同時把翻譯模型暖機） ----
Write-Info "啟動 LiveTranslate（第一次載入語音模型約需 5-10 秒）..."
Write-Info "關閉方式：右下角系統匣圖示 -> 結束，或字幕視窗右上角「結束」。"
Set-Location $Root
$app = Start-Process -FilePath $Python -ArgumentList "main.py" -WorkingDirectory $Root -NoNewWindow -PassThru

# 暖機：載入模型並真的跑一次翻譯，確保第一句不會等待模型載入而逾時
Write-Info "翻譯模型暖機中..."
try {
    $body = '{"model":"' + $Model + '","keep_alive":"30m"}'
    Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/generate" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 180 | Out-Null
    $chat = '{"model":"' + $Model + '","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"reasoning_effort":"none"}'
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType "application/json" -Body $chat -TimeoutSec 120 | Out-Null
    Write-Ok ("翻譯模型就緒（第一次回應 {0} ms）。" -f [int]$sw.ElapsedMilliseconds)
} catch {
    Write-Warn "暖機失敗：$($_.Exception.Message)（程式仍會啟動，第一句翻譯可能較慢）"
}

$app.WaitForExit()
$code = $app.ExitCode

# ---- Step 5: release the GPU ----
Write-Info "釋放顯示記憶體：卸載翻譯模型並關閉本腳本啟動的 Ollama..."
try {
    $ps = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/ps" -TimeoutSec 3
    foreach ($m in @($ps.models)) {
        $body = '{"model":"' + $m.name + '","keep_alive":0}'
        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/generate" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 15 | Out-Null
        Write-Ok "已卸載 $($m.name)"
    }
} catch {}
if (Test-Path $PidFile) {
    try {
        $opid = [int](Get-Content $PidFile -ErrorAction Stop)
        $proc = Get-Process -Id $opid -ErrorAction Stop
        if ($proc.Name -eq "ollama") { Stop-Process -Id $opid -Force; Write-Ok "Ollama 服務已關閉（下次啟動會自動重開）。" }
    } catch {}
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}
if ($code -ne 0) {
    Write-Err "LiveTranslate 異常結束（exit code $code）。"
    Read-Host "按 Enter 關閉"
}
