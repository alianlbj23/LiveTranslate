# 顯示記憶體：安裝時依顯示卡選預設模型、啟動前檢查可用量。
# 判斷都在 hardware_profile.py（經 launcher\hardware_advice.py），這裡只負責呼叫與顯示。
# 由 setup.ps1、start_jp_subtitles.ps1、gui.ps1 dot-source；也可單獨測試。

function Get-FreeVramMB {
    # 回傳第 $Index 張顯示卡的可用記憶體（MB）；沒有 nvidia-smi 或讀不到時回傳 -1
    param([int]$Index = 0)
    try {
        $line = (& nvidia-smi "--id=$Index" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        if (-not $line) { return -1 }
        $parts = "$line" -split ","
        return [int]($parts[1].Trim()) - [int]($parts[0].Trim())
    } catch {
        Write-Host "[注意] 讀不到可用顯示記憶體：$($_.Exception.Message)" -ForegroundColor DarkGray
        return -1
    }
}

function Invoke-HardwareAdvice {
    # 執行 launcher\hardware_advice.py，回傳解析後的 JSON；失敗或超過 $TimeoutMs 回傳 $null
    # （原因印在畫面上，呼叫端改用原本的做法）。讀顯示卡（NVML）卡住時也不會擋住啟動視窗。
    param([string]$Python, [string]$Root, [string[]]$Arguments, [int]$TimeoutMs = 20000)
    $helper = Join-Path $Root "launcher\hardware_advice.py"
    if (-not (Test-Path -LiteralPath $Python) -or -not (Test-Path -LiteralPath $helper)) { return $null }
    $proc = $null
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Python
        $psi.Arguments = (@("-I", $helper) + $Arguments | ForEach-Object { '"' + ([string]$_).Replace('"', '\"') + '"' }) -join " "
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
        $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
        $proc = [System.Diagnostics.Process]::Start($psi)
        $outTask = $proc.StandardOutput.ReadToEndAsync()
        $errTask = $proc.StandardError.ReadToEndAsync()
        if (-not $proc.WaitForExit($TimeoutMs)) {
            $proc.Kill()
            Write-Host ("[注意] 判斷顯示卡超過 {0} 秒沒有回應，略過建議。" -f [int]($TimeoutMs / 1000)) -ForegroundColor DarkGray
            return $null
        }
        $proc.WaitForExit()  # let the output readers finish
        $err = $errTask.Result.Trim()
        if ($err) { Write-Host $err -ForegroundColor DarkGray }
        if ($proc.ExitCode -ne 0) {
            Write-Host "[注意] 無法判斷顯示卡能跑哪些模型（exit code $($proc.ExitCode)）。" -ForegroundColor DarkGray
            return $null
        }
        $line = @($outTask.Result.Trim() -split "`r?`n")[-1]
        return ($line | ConvertFrom-Json)
    } catch {
        Write-Host "[注意] 無法判斷顯示卡能跑哪些模型：$($_.Exception.Message)" -ForegroundColor DarkGray
        return $null
    } finally {
        if ($proc) { $proc.Dispose() }
    }
}

function Get-CatalogDefaultModel {
    # local_models.json 的 default；讀不到時用 qwen3.5:9b
    param([string]$Root)
    try {
        $data = Get-Content -LiteralPath (Join-Path $Root "local_models.json") -Raw -Encoding UTF8 | ConvertFrom-Json
        $hit = @($data.models | Where-Object { $_.model -eq $data.default })
        if ($hit.Count -gt 0) { return $hit[0] }
    } catch {
        Write-Host "[注意] 讀不到 local_models.json：$($_.Exception.Message)" -ForegroundColor DarkGray
    }
    return [pscustomobject]@{ model = "qwen3.5:9b"; label = "Qwen3.5 9B"; size_gb = 6.6 }
}

function Get-InstallChoice {
    # 第一次安裝：依顯示卡總記憶體（MB，來自 nvidia-smi）選預設的辨識＋翻譯模型。
    # 回傳 asr_choice（whisper-turbo / sensevoice）、model、label、size_gb、level、message；
    # 判斷失敗就用 Whisper turbo＋目錄的預設翻譯模型（原本的做法）。
    param([string]$Python, [string]$Root, [int]$TotalMB)
    $c = Invoke-HardwareAdvice -Python $Python -Root $Root -Arguments @("install", "--total-mb", "$TotalMB")
    if ($c) { return $c }
    $d = Get-CatalogDefaultModel -Root $Root
    return [pscustomobject]@{ asr_choice = "whisper-turbo"; model = $d.model; label = $d.label; size_gb = $d.size_gb
        level = "unknown"; total_need_gb = $null; message = "無法判斷顯示卡，使用預設的 Whisper turbo＋$($d.label)。" }
}

function Get-StartupVramNeed {
    # 啟動前：目前的辨識模型＋這次要用的翻譯模型，需要多少可用顯示記憶體
    # （free_need_mb、asr_label、model_label、gpu_index）。判斷失敗回傳 $null，呼叫端改用固定門檻。
    param([string]$Python, [string]$Root, [string]$Settings, [string]$Model)
    return Invoke-HardwareAdvice -Python $Python -Root $Root -Arguments @("startup", $Settings, $Model)
}

function Get-OllamaLoadedVramMB {
    # Ollama 目前佔用的顯示記憶體（MB，/api/ps 的 size_vram 總和）。這些模型會被換成這次要用的模型
    # （OLLAMA_MAX_LOADED_MODELS=1），所以算在可用量裡；讀不到回傳 0。
    param([int]$Port)
    try {
        $ps = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/ps" -TimeoutSec 3
        $sum = 0.0
        foreach ($m in @($ps.models)) { if ($m.size_vram) { $sum += [double]$m.size_vram } }
        return [int]($sum / 1MB)
    } catch {
        Write-Host "[注意] 讀不到 Ollama 已載入的模型：$($_.Exception.Message)" -ForegroundColor DarkGray
        return 0
    }
}

function Test-LowVram {
    # 可用量低於門檻就印出提示並回傳 $true；-1（無法取得）視為正常。$Pair = 「辨識模型＋翻譯模型」的名稱
    param([int]$FreeMB, [int]$ThresholdMB = 8000, [string]$Pair = "")
    if ($FreeMB -lt 0 -or $FreeMB -ge $ThresholdMB) { return $false }
    $what = if ($Pair) { "目前的組合（$Pair）" } else { "辨識＋翻譯模型" }
    Write-Host ("[注意] 可用顯示記憶體只有 {0:N1} GB，{1}大約需要 {2:N1} GB；不夠時翻譯會變得極慢甚至像卡住。" -f `
        ($FreeMB / 1024), $what, ($ThresholdMB / 1024)) -ForegroundColor Yellow
    Write-Host "       建議：關閉 Wallpaper Engine、遊戲、多餘的瀏覽器分頁；或在啟動視窗改用標示【推薦】的翻譯模型。" -ForegroundColor Yellow
    return $true
}
