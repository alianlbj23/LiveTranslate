# 安裝時依顯示卡選預設的辨識＋翻譯模型（setup.ps1 dot-source）。
# 判斷都在 hardware_profile.py（經 launcher\hardware_advice.py），這裡只負責呼叫。
# 啟動時的顯示記憶體檢查已經搬進 LiveTranslate（local_model_prep.py）。

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
