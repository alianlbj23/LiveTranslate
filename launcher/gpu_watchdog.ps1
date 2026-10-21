# 顯示卡看門狗：LiveTranslate 啟動 Ollama 時（ollama_server.start_watchdog）以隱藏視窗啟動。
# 等 LiveTranslate 結束（正常結束、當掉都算）後：卸載 Ollama 裡所有模型，並關閉它開的 ollama serve。
# 正常結束時 LiveTranslate 已先清過，這裡會發現沒事做；所有動作都是可重複的。
param(
    [Parameter(Mandatory = $true)][int]$ParentPid,
    [Parameter(Mandatory = $true)][int]$Port,
    [Parameter(Mandatory = $true)][string]$PidFile,
    [Parameter(Mandatory = $true)][string]$LogFile
)
$ErrorActionPreference = "Continue"

function Log {
    param($m)
    $line = "{0} [watchdog pid={1}] {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $PID, $m
    try { Add-Content -Path $LogFile -Value $line -Encoding UTF8 } catch {}
}

Log "waiting for launcher pid=$ParentPid (ollama port $Port)"
try {
    Wait-Process -Id $ParentPid -ErrorAction Stop
} catch {
    Log "launcher already gone: $($_.Exception.Message)"
}
Start-Sleep -Seconds 2
Log "launcher exited, cleaning up"

# 1) 卸載模型（Ollama 可能還在跑）
try {
    $ps = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/ps" -TimeoutSec 3
    $models = @($ps.models)
    if ($models.Count -eq 0) { Log "no model loaded" }
    foreach ($m in $models) {
        $body = '{"model":"' + $m.name + '","keep_alive":0}'
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/generate" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 20 | Out-Null
            Log "unloaded $($m.name)"
        } catch {
            Log "unload $($m.name) failed: $($_.Exception.Message)"
        }
    }
} catch {
    Log "ollama not answering on $Port (already stopped?): $($_.Exception.Message)"
}

# 2) 關閉啟動腳本自己開的 ollama serve（只依 pid 檔，不動別人的 Ollama）
if (Test-Path $PidFile) {
    try {
        $opid = [int](Get-Content $PidFile -ErrorAction Stop)
        $proc = Get-Process -Id $opid -ErrorAction Stop
        if ($proc.Name -eq "ollama") {
            Stop-Process -Id $opid -Force
            Log "stopped ollama serve pid=$opid"
        } else {
            Log "pid $opid is '$($proc.Name)', not ollama; left alone"
        }
    } catch {
        Log "ollama serve pid from pid file not running: $($_.Exception.Message)"
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
} else {
    Log "no pid file (launcher already cleaned up)"
}
Log "done"
exit 0
