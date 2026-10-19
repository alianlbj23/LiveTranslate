# 驗證 GPU / faster-whisper CUDA / Ollama 是否正常
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Root   = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
. (Join-Path $Root "launcher\model.ps1")
$Model  = Get-LauncherModel -Path (Join-Path $Root "launcher\selected_model.txt")

Write-Host ""
Write-Host "=== 1. 顯示卡驅動 ===" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv

Write-Host ""
Write-Host "=== 2. PyTorch CUDA ===" -ForegroundColor Cyan
& $Python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

Write-Host ""
Write-Host "=== 3. faster-whisper / CTranslate2 CUDA ===" -ForegroundColor Cyan
& $Python -c "import torch, ctranslate2, faster_whisper; print('ctranslate2', ctranslate2.__version__, '| faster-whisper', faster_whisper.__version__, '| cuda devices:', ctranslate2.get_cuda_device_count(), '| compute types:', sorted(ctranslate2.get_supported_compute_types('cuda')))"

Write-Host ""
Write-Host "=== 4. Ollama ===" -ForegroundColor Cyan
foreach ($port in 11434, 11435) {
    try {
        $v = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/version" -TimeoutSec 2
        $owner = "unknown"
        try { $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop | Select-Object -First 1; $owner = (Get-Process -Id $c.OwningProcess).Name } catch {}
        $tags = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/tags" -TimeoutSec 3
        $names = ($tags.models | ForEach-Object { $_.name }) -join ", "
        Write-Host "連接埠 $port : Ollama $($v.version)（程序 $owner）模型：$names"
    } catch {
        Write-Host "連接埠 $port : 沒有 Ollama 回應" -ForegroundColor DarkGray
    }
}
if (Test-Path $Ollama) { & $Ollama --version } else { Write-Host "找不到 Windows 版 Ollama：$Ollama" -ForegroundColor Yellow }

Write-Host ""
Write-Host "=== 5. 翻譯測試（$Model，關閉思考模式）===" -ForegroundColor Cyan
$port = if (Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction SilentlyContinue) { 11435 } else { 11434 }
try {
    $body = @{ model = $Model; stream = $false; think = $false; messages = @(@{ role = "user"; content = "請把下面這句日文翻譯成台灣繁體中文，只輸出翻譯：今日はいい天気ですね" }) } | ConvertTo-Json -Depth 5
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/chat" -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 180
    $sw.Stop()
    Write-Host "翻譯結果：$($r.message.content)  （$([int]$sw.Elapsed.TotalMilliseconds) ms，thinking 內容長度：$(([string]$r.message.thinking).Length)）"
} catch {
    Write-Host "翻譯測試失敗：$($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""
