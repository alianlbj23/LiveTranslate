# LiveTranslate 單一入口選單
# 雙擊 LiveTranslate.bat 會開這個選單；直接按 Enter = 第 1 項（啟動雙字幕），輸入數字進其他功能。
# 也可直接指定動作（捷徑用）：powershell -File menu.ps1 -Action start | verify
param(
    [ValidateSet("menu", "start", "verify", "setup")]
    [string]$Action = "menu"
)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Root     = Split-Path -Parent $PSScriptRoot   # launcher\ 的上一層，搬資料夾也不用改
$Python   = Join-Path $Root ".venv\Scripts\python.exe"
$Ollama   = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
$Settings = Join-Path $Root "user_settings.json"
$Helper   = Join-Path $Root "launcher\set_api_base.py"
$StartScript  = Join-Path $Root "launcher\start_jp_subtitles.ps1"
$VerifyScript = Join-Path $Root "launcher\verify.ps1"
$SetupScript  = Join-Path $Root "launcher\setup.ps1"
$ReadyMarker  = Join-Path $Root ".venv\.livetranslate-ready"

function Get-ActivePort {
    if (Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction SilentlyContinue) { return 11435 }
    return 11434
}

function Switch-Model {
    param([string]$Model)
    $port = Get-ActivePort
    $alive = $false
    try { Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/version" -TimeoutSec 2 | Out-Null; $alive = $true } catch {}
    if (-not $alive) {
        Write-Host "Ollama 沒有在執行，先啟動 Windows 版 Ollama..." -ForegroundColor Yellow
        $env:OLLAMA_HOST = "127.0.0.1:$port"; $env:OLLAMA_KEEP_ALIVE = "30m"; $env:OLLAMA_FLASH_ATTENTION = "1"; $env:OLLAMA_KV_CACHE_TYPE = "q8_0"; $env:OLLAMA_NUM_PARALLEL = "1"; $env:OLLAMA_CONTEXT_LENGTH = "2048"
        $p = Start-Process -FilePath $Ollama -ArgumentList "serve" -WindowStyle Hidden -PassThru; Set-Content -Path (Join-Path $Root "launcher\ollama.pid") -Value $p.Id -Encoding ascii
        Start-Sleep -Seconds 5
    }
    $env:OLLAMA_HOST = "127.0.0.1:$port"
    Write-Host "下載 / 確認模型 $Model ..." -ForegroundColor Cyan
    & $Ollama pull $Model
    if ($LASTEXITCODE -ne 0) { Write-Host "下載失敗。" -ForegroundColor Red; return }
    if (Test-Path $Settings) {
        & $Python -I $Helper $Settings "http://127.0.0.1:$port/v1" $Model
    }
    # 讓啟動腳本之後也用這個模型
    $txt = [IO.File]::ReadAllText($StartScript, [Text.Encoding]::UTF8)
    $txt = $txt -replace '\$Model\s*=\s*"qwen3\.5:\d+b"', ('$Model    = "' + $Model + '"')
    [IO.File]::WriteAllText($StartScript, $txt, (New-Object Text.UTF8Encoding $true))
    Write-Host "已切換為 $Model（LiveTranslate 重新啟動後生效）。" -ForegroundColor Green
}

function Invoke-Choice {
    param([string]$Choice)
    switch ($Choice.Trim()) {
        "1" { & $StartScript }
        "2" { & $VerifyScript }
        "3" { Switch-Model "qwen3.5:4b" }
        "4" { Switch-Model "qwen3.5:9b" }
        "5" { Start-Process (Join-Path $Root "INSTALL_GUIDE.md") }
        "0" { exit 0 }
        ""  { & $StartScript }
        default { Write-Host "無效的選項：$Choice" -ForegroundColor Yellow }
    }
}

if ($Action -eq "setup")  { & $SetupScript;  exit $LASTEXITCODE }

# 第一次使用（沒有 .venv 或安裝沒跑完）：先自動安裝
if (-not (Test-Path $Python) -or -not (Test-Path $ReadyMarker)) {
    Write-Host "尚未安裝（找不到 .venv 或安裝未完成），開始一鍵安裝..." -ForegroundColor Yellow
    & $SetupScript
    if ($LASTEXITCODE -ne 0) { exit 1 }
    if (-not (Test-Path $Python)) { exit 1 }
}

if ($Action -eq "start")  { & $StartScript;  exit $LASTEXITCODE }
if ($Action -eq "verify") { & $VerifyScript; exit $LASTEXITCODE }

while ($true) {
    Write-Host ""
    Write-Host "============ LiveTranslate ============" -ForegroundColor Magenta
    Write-Host " 1) 啟動日文雙字幕                （預設：直接按 Enter）"
    Write-Host " 2) 檢查 GPU / Ollama / 翻譯是否正常"
    Write-Host " 3) 切換翻譯模型為 qwen3.5:4b（顯示記憶體不足時用）"
    Write-Host " 4) 切換翻譯模型為 qwen3.5:9b（預設，品質較好）"
    Write-Host " 5) 開啟安裝說明 INSTALL_GUIDE.md"
    Write-Host "    （每次啟動會自動檢查更新；按「結束」關閉程式就會釋放顯示卡）" -ForegroundColor DarkGray
    Write-Host " 0) 離開"
    $choice = Read-Host "請輸入數字（直接按 Enter = 啟動）"
    Invoke-Choice $choice
}
