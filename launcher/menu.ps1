# LiveTranslate 單一入口選單
# 雙擊 LiveTranslate.bat 會開啟動視窗（launcher/gui.ps1）：Enter = 啟動字幕；也能切換翻譯模型、檢查環境。
# 視窗開不了時（例如沒有桌面環境）自動改用文字選單：直接按 Enter = 第 1 項，輸入數字進其他功能。
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
$ModelFile    = Join-Path $Root "launcher\selected_model.txt"
$GuideFile    = Join-Path $Root "INSTALL_GUIDE.md"
. (Join-Path $Root "launcher\model.ps1")
. (Join-Path $Root "launcher\gui.ps1")
. (Join-Path $Root "launcher\process.ps1")
. (Join-Path $Root "launcher\bench.ps1")

function Get-ActivePort {
    if (Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction SilentlyContinue) { return 11435 }
    return 11434
}

function Switch-Model {
    # 下載（需要時）並切換翻譯模型；成功回傳 $true。外部程式的輸出一律送到畫面（Out-Host），不混進回傳值。
    param([string]$Model)
    $port = Get-ActivePort
    $alive = $false
    try { Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/version" -TimeoutSec 2 | Out-Null; $alive = $true }
    catch { $alive = $false }  # expected when Ollama is not running yet: start it below
    if (-not $alive) {
        Write-Host "Ollama 沒有在執行，先啟動 Windows 版 Ollama..." -ForegroundColor Yellow
        $env:OLLAMA_HOST = "127.0.0.1:$port"; $env:OLLAMA_KEEP_ALIVE = "30m"; $env:OLLAMA_FLASH_ATTENTION = "1"; $env:OLLAMA_KV_CACHE_TYPE = "q8_0"; $env:OLLAMA_NUM_PARALLEL = "2"; $env:OLLAMA_CONTEXT_LENGTH = "2048"; $env:OLLAMA_MAX_LOADED_MODELS = "1"
        $p = Start-Process -FilePath $Ollama -ArgumentList "serve" -WindowStyle Hidden -PassThru; Set-Content -Path (Join-Path $Root "launcher\ollama.pid") -Value $p.Id -Encoding ascii
        Start-Sleep -Seconds 5
    }
    $env:OLLAMA_HOST = "127.0.0.1:$port"
    if (Test-OllamaModelDownloaded -Model $Model) {
        # already on disk: no registry round trip, so switching also works offline
        Write-Host "模型 $Model 已下載。" -ForegroundColor Cyan
    } else {
        Write-Host "下載模型 $Model ..." -ForegroundColor Cyan
        & $Ollama pull $Model | Out-Host
        if ($LASTEXITCODE -ne 0) { Write-Host "下載失敗（exit code $LASTEXITCODE）。" -ForegroundColor Red; return $false }
    }
    if (Test-Path $Settings) {
        & $Python -I $Helper $Settings "http://127.0.0.1:$port/v1" $Model | Out-Host
        if ($LASTEXITCODE -ne 0) { Write-Host "更新設定檔失敗（exit code $LASTEXITCODE），沒有切換。" -ForegroundColor Red; return $false }
    }
    # 讓啟動腳本之後也用這個模型（存在 git 不追蹤的 selected_model.txt，不改腳本本身）
    if (-not (Set-LauncherModel -Path $ModelFile -Model $Model)) { return $false }
    Write-Host "已切換為 $Model（LiveTranslate 重新啟動後生效）。" -ForegroundColor Green
    return $true
}

function Show-TextMenu {
    # 文字選單（啟動視窗開不了時的備援）：模型清單來自 local_models.json
    $catalog = @(Get-LauncherCatalog -Root $Root)
    $current = Get-LauncherModel -Path $ModelFile
    Write-Host ""
    Write-Host "============ LiveTranslate ============" -ForegroundColor Magenta
    Write-Host " 1) 啟動日文雙字幕                （預設：直接按 Enter）"
    Write-Host " 2) 檢查 GPU / Ollama / 翻譯是否正常"
    for ($i = 0; $i -lt $catalog.Count; $i++) {
        $e = $catalog[$i]
        $mark = if ($e.model -eq $current) { "（使用中）" } else { "" }
        Write-Host (" {0}) 切換翻譯模型為 {1}：{2}{3}" -f ($i + 3), $e.label, $e.note, $mark)
    }
    $guideNo = $catalog.Count + 3
    Write-Host (" {0}) 開啟安裝說明 INSTALL_GUIDE.md" -f $guideNo)
    $benchNo = $guideNo + 1
    Write-Host (" {0}) 最佳組合評測（找出最好的辨識＋翻譯組合）" -f $benchNo)
    Write-Host "    （每次啟動會自動檢查更新；按「結束」關閉程式就會釋放顯示卡）" -ForegroundColor DarkGray
    Write-Host " 0) 離開"
    $choice = (Read-Host "請輸入數字（直接按 Enter = 啟動）").Trim()
    $n = 0
    if ($choice -eq "" -or $choice -eq "1") { return @{ Action = "start" } }
    if ($choice -eq "2") { return @{ Action = "verify" } }
    if ($choice -eq "0") { return @{ Action = "exit" } }
    if ([int]::TryParse($choice, [ref]$n)) {
        if ($n -eq $guideNo) { return @{ Action = "guide" } }
        if ($n -eq $benchNo) { return @{ Action = "bench" } }
        if ($n -ge 3 -and $n -lt $guideNo) {
            $e = $catalog[$n - 3]
            $ask = { param($text) ((Read-Host ($text.Replace("`n`n", " ") + " (y/N)")).Trim().ToLower() -eq "y") }
            $decision = Resolve-SwitchRequest -Entry $e -CurrentModel $current `
                -Downloaded (Test-OllamaModelDownloaded -Model $e.model) -Confirm $ask
            if ($decision -eq "switch") { return @{ Action = "switch"; Model = $e.model } }
            if ($e.model -eq $current) { Write-Host "目前已經在使用 $($e.label)。" -ForegroundColor Green }
            return @{ Action = "none" }
        }
    }
    Write-Host "無效的選項：$choice" -ForegroundColor Yellow
    return @{ Action = "none" }
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

$useText = $false
while ($true) {
    $r = $null
    if (-not $useText) {
        try {
            $r = Show-LauncherWindow -Root $Root -CurrentModel (Get-LauncherModel -Path $ModelFile)
        } catch {
            Write-Host "[注意] 啟動視窗無法開啟（$($_.Exception.Message)），改用文字選單。" -ForegroundColor Yellow
            $useText = $true
        }
    }
    if ($useText) { $r = Show-TextMenu }
    switch ($r.Action) {
        "start"  {
            # 視窗裡選了別的模型：先換（必要時下載），換不成功就不啟動，免得默默沿用舊模型
            $ok = $true
            if ($r.Model) { $ok = Switch-Model $r.Model }
            if ($ok) { & $StartScript } else { [void](Read-Host "模型沒有切換成功，按 Enter 回到選單") }
        }
        "verify" { & $VerifyScript; [void](Read-Host "按 Enter 回到選單") }
        "guide"  { Start-Process $GuideFile }
        "bench"  { Show-BenchMenu -Root $Root -Python $Python }
        "switch" { [void](Switch-Model $r.Model); [void](Read-Host "按 Enter 回到選單") }
        "none"   { }
        default  { exit 0 }
    }
}
