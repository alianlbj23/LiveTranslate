# 顯示記憶體檢查（由 start_jp_subtitles.ps1 dot-source；也可單獨測試）
function Get-FreeVramMB {
    # 回傳可用顯示記憶體（MB）；沒有 nvidia-smi 時回傳 -1
    try {
        $line = (& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        if (-not $line) { return -1 }
        $parts = "$line" -split ","
        return [int]($parts[1].Trim()) - [int]($parts[0].Trim())
    } catch {
        return -1
    }
}

function Test-LowVram {
    # 低於門檻就印出提示並回傳 $true；-1（無法取得）視為正常
    param([int]$FreeMB, [int]$ThresholdMB = 8000)
    if ($FreeMB -lt 0 -or $FreeMB -ge $ThresholdMB) { return $false }
    Write-Host ("[注意] 可用顯示記憶體只有 {0:N1} GB（建議至少 {1:N0} GB）。翻譯模型要 5.6 GB、語音模型 1.5 GB，不夠時翻譯會變得極慢甚至像卡住。" -f ($FreeMB / 1024), ($ThresholdMB / 1024)) -ForegroundColor Yellow
    Write-Host "       建議：關閉 Wallpaper Engine、遊戲、多餘的 Chrome 分頁；或在選單按 3 改用 qwen3.5:4b。" -ForegroundColor Yellow
    return $true
}
