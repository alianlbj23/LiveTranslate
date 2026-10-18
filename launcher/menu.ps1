# LiveTranslate 維護選單
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Root     = "D:\workspace\language_translate"
$Python   = Join-Path $Root ".venv\Scripts\python.exe"
$Ollama   = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
$Settings = Join-Path $Root "user_settings.json"
$Helper   = Join-Path $Root "launcher\set_api_base.py"

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
        $env:OLLAMA_HOST = "127.0.0.1:$port"; $env:OLLAMA_KEEP_ALIVE = "30m"; $env:OLLAMA_FLASH_ATTENTION = "1"; $env:OLLAMA_KV_CACHE_TYPE = "q8_0"
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
    $launcher = Join-Path $Root "launcher\start_jp_subtitles.ps1"
    $txt = [IO.File]::ReadAllText($launcher, [Text.Encoding]::UTF8)
    $txt = $txt -replace '\$Model\s*=\s*"qwen3\.5:\d+b"', ('$Model    = "' + $Model + '"')
    [IO.File]::WriteAllText($launcher, $txt, (New-Object Text.UTF8Encoding $true))
    Write-Host "已切換為 $Model（LiveTranslate 重新啟動後生效）。" -ForegroundColor Green
}

while ($true) {
    Write-Host ""
    Write-Host "============ LiveTranslate 維護選單 ============" -ForegroundColor Magenta
    Write-Host " 1) 啟動日文雙字幕"
    Write-Host " 2) 檢查 GPU / Ollama / 翻譯是否正常"
    Write-Host " 3) 切換翻譯模型為 qwen3.5:4b（顯示記憶體不足時用）"
    Write-Host " 4) 切換翻譯模型為 qwen3.5:9b（預設，品質較好）"
    Write-Host " 5) 更新 LiveTranslate（git pull + 更新套件）"
    Write-Host " 6) 停止 Ollama（釋放顯示記憶體，玩遊戲前一定要按）"
    Write-Host " 7) 開啟安裝說明 INSTALL_GUIDE.md"
    Write-Host " 0) 離開"
    $choice = Read-Host "請輸入數字"
    switch ($choice) {
        "1" { & (Join-Path $Root "launcher\start_jp_subtitles.ps1") }
        "2" { & (Join-Path $Root "launcher\verify.ps1") }
        "3" { Switch-Model "qwen3.5:4b" }
        "4" { Switch-Model "qwen3.5:9b" }
        "5" { Push-Location $Root; cmd /c update.bat; Pop-Location }
        "6" {
            Get-Process -Name "ollama", "ollama app" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
            Write-Host "已停止 Windows 版 Ollama。" -ForegroundColor Green
        }
        "7" { Start-Process (Join-Path $Root "INSTALL_GUIDE.md") }
        "0" { exit 0 }
        default { Write-Host "無效的選項" -ForegroundColor Yellow }
    }
}
