# 更新視窗：LiveTranslate 關閉之後，安裝下載好的新版本（-Zip，安裝包版本）、必要時重新安裝套件
# （requirements.txt 有變時），再重新啟動 LiveTranslate。
# 由 app_update.launch_restart 以隱藏的主控台啟動，畫面上只有這個小視窗；過程同時寫進 logs\update_*.log。
param(
    [int]$ParentPid = 0,
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    [string]$Zip = "",        # 新版本的 zip（內容在 LiveTranslate\ 底下），複製到程式資料夾
    [string]$StartedFile = "",  # 寫入這個檔表示視窗已經起來（LiveTranslate 等到它才結束）
    [switch]$Packages
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.Windows.Forms.Application]::EnableVisualStyles()
if ($StartedFile) { Set-Content -LiteralPath $StartedFile -Value $PID -Encoding ascii }

$Python  = Join-Path $Root ".venv\Scripts\python.exe"
$PythonW = Join-Path $Root ".venv\Scripts\pythonw.exe"
$Ready   = Join-Path $Root ".venv\.livetranslate-ready"
$LogDir  = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$script:LogFile = Join-Path $LogDir ("update_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
$WaitSeconds = 120
$StepTimeoutMinutes = 60          # a pip step that hangs longer than this is ended
# pip cache and temp files inside the app folder (not the system drive), deleted afterwards
$PipCache = Join-Path $Root ".pip-cache"
$PipTemp  = Join-Path $Root ".tmp"

$Steps = @(
    @{ Title = "更新套件（requirements.txt）"; Arguments = @("-m", "pip", "install", "-r", "requirements.txt", "--progress-bar", "off") },
    @{ Title = "更新斷句套件（yasbd-lib）"; Arguments = @("-m", "pip", "install", "yasbd-lib>=0.15,<1.0", "--progress-bar", "off") },
    @{ Title = "檢查套件相依性"; Arguments = @("-m", "pip", "check") }
)

# ---------- window ----------
$form = New-Object System.Windows.Forms.Form
$form.Text = "LiveTranslate 更新"
$form.ClientSize = New-Object System.Drawing.Size(560, 350)
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.TopMost = $true
$form.Font = New-Object System.Drawing.Font("Microsoft JhengHei UI", 9.5)

$label = New-Object System.Windows.Forms.Label
$label.Location = New-Object System.Drawing.Point(16, 14)
$label.Size = New-Object System.Drawing.Size(528, 52)
$label.Text = "等待 LiveTranslate 關閉…"
$form.Controls.Add($label)

$box = New-Object System.Windows.Forms.TextBox
$box.Location = New-Object System.Drawing.Point(16, 70)
$box.Size = New-Object System.Drawing.Size(528, 222)
$box.Multiline = $true
$box.ReadOnly = $true
$box.ScrollBars = "Vertical"
$box.Font = New-Object System.Drawing.Font("Microsoft JhengHei UI", 8.5)  # Consolas has no Chinese
$form.Controls.Add($box)

$btnStart = New-Object System.Windows.Forms.Button
$btnStart.Text = "仍然啟動 LiveTranslate"
$btnStart.Location = New-Object System.Drawing.Point(260, 304)
$btnStart.Size = New-Object System.Drawing.Size(180, 32)
$btnStart.Visible = $false
$form.Controls.Add($btnStart)

$btnClose = New-Object System.Windows.Forms.Button
$btnClose.Text = "關閉"
$btnClose.Location = New-Object System.Drawing.Point(450, 304)
$btnClose.Size = New-Object System.Drawing.Size(94, 32)
$btnClose.Enabled = $false
$btnClose.Add_Click({ $form.Close() })
$form.Controls.Add($btnClose)

# ---------- helpers ----------
function Add-Line {
    param([string]$Text)
    $box.AppendText($Text + "`r`n")
    try { Add-Content -LiteralPath $script:LogFile -Value $Text -Encoding UTF8 } catch {
        $box.AppendText("（寫不進記錄檔：$($_.Exception.Message)）`r`n")
    }
}

function Set-Failed {
    # -NoStart: LiveTranslate may still be running, so no "start anyway" (two copies would run)
    param([string]$Text, [switch]$NoStart)
    $script:State = "failed"
    $label.ForeColor = [System.Drawing.Color]::Firebrick
    $label.Text = "$Text`r`n詳細內容在下方。"
    Add-Line "[失敗] $Text"
    Add-Line "記錄檔：$script:LogFile"
    $btnStart.Visible = (-not $NoStart) -and (Test-Path -LiteralPath $PythonW)
    $btnClose.Enabled = $true
}

function Start-LiveTranslate {
    $label.Text = "重新啟動 LiveTranslate…"
    Add-Line "啟動 LiveTranslate"
    $script:AppProc = Start-Process -FilePath $PythonW -ArgumentList "main.py" -WorkingDirectory $Root -PassThru
}

function Start-Step {
    param($Step)
    $label.Text = $Step.Title + "（可能需要幾分鐘）…"
    Add-Line ("> " + $Step.Title)
    New-Item -ItemType Directory -Force -Path $PipCache, $PipTemp | Out-Null
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Python
    $psi.Arguments = ($Step.Arguments | ForEach-Object { '"' + ([string]$_).Replace('"', '\"') + '"' }) -join " "
    $psi.WorkingDirectory = $Root
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8"
    $psi.EnvironmentVariables["PIP_CACHE_DIR"] = $PipCache
    $psi.EnvironmentVariables["TMP"] = $PipTemp
    $psi.EnvironmentVariables["TEMP"] = $PipTemp
    $script:Proc = [System.Diagnostics.Process]::Start($psi)
    $script:StepStarted = Get-Date
    $script:OutTask = $script:Proc.StandardOutput.ReadLineAsync()
    $script:ErrTask = $script:Proc.StandardError.ReadLineAsync()
}

function Update-StepOutput {
    # Show the lines that arrived; returns $true while the step is still running
    while ($script:OutTask -and $script:OutTask.IsCompleted) {
        $line = $script:OutTask.Result
        if ($null -eq $line) { $script:OutTask = $null } else { Add-Line $line; $script:OutTask = $script:Proc.StandardOutput.ReadLineAsync() }
    }
    while ($script:ErrTask -and $script:ErrTask.IsCompleted) {
        $line = $script:ErrTask.Result
        if ($null -eq $line) { $script:ErrTask = $null } else { Add-Line $line; $script:ErrTask = $script:Proc.StandardError.ReadLineAsync() }
    }
    return -not ($script:Proc.HasExited -and -not $script:OutTask -and -not $script:ErrTask)
}

function Stop-StepTree {
    # pip may run other programs below it: end the whole tree
    if (-not $script:Proc -or $script:Proc.HasExited) { return }
    try {
        $k = Start-Process -FilePath "taskkill.exe" -ArgumentList "/T", "/F", "/PID", $script:Proc.Id -WindowStyle Hidden -PassThru
        if (-not $k.WaitForExit(10000)) { Add-Line "taskkill 沒有在 10 秒內結束" }
    } catch {
        Add-Line "無法結束安裝套件的程式：$($_.Exception.Message)"
    }
}

function Start-PackageSteps {
    Remove-Item -LiteralPath $Ready -Force -ErrorAction SilentlyContinue
    $script:State = "steps"
    Start-Step $Steps[0]
}

function Install-Release {
    # Copy the downloaded release over the program files. The zip holds only program files,
    # so .venv, models, user_settings.json, logs and transcripts are not touched.
    $label.Text = "安裝新版本…"
    Add-Line "> 安裝新版本：$Zip"
    $work = Join-Path $Root ".update\new"
    if (Test-Path -LiteralPath $work) { Remove-Item -Recurse -Force -LiteralPath $work }
    # .NET refuses entries that would land outside $work (zip slip)
    [System.IO.Compression.ZipFile]::ExtractToDirectory($Zip, $work)
    $src = Join-Path $work "LiveTranslate"
    if (-not (Test-Path -LiteralPath (Join-Path $src "main.py"))) {
        Set-Failed "下載的新版本不完整，沒有更新（目前的版本仍可使用）。"
        return $false
    }
    # config.yaml is in the release too: keep the user's edits in a copy
    $config = Join-Path $Root "config.yaml"
    $newConfig = Join-Path $src "config.yaml"
    if ((Test-Path -LiteralPath $config) -and (Test-Path -LiteralPath $newConfig) -and
            ((Get-FileHash -LiteralPath $config).Hash -ne (Get-FileHash -LiteralPath $newConfig).Hash)) {
        $backup = Join-Path $Root ("config.backup_{0:yyyyMMdd_HHmmss}.yaml" -f (Get-Date))
        Copy-Item -LiteralPath $config -Destination $backup
        Add-Line "原本的 config.yaml 備份在 $backup"
    }
    # From here a half-copied folder must not look ready: LiveTranslate.exe then runs the
    # installer again instead of starting a mix of old and new files
    Remove-Item -LiteralPath $Ready -Force -ErrorAction SilentlyContinue
    $out = & robocopy.exe $src $Root /E /R:2 /W:1 /NFL /NDL /NJH /NJS /NP
    $code = $LASTEXITCODE
    foreach ($l in @($out)) { if ("$l".Trim()) { Add-Line "$l".Trim() } }
    if ($code -ge 8) {  # robocopy: 0-7 = copied / nothing to copy, 8+ = failures
        $msg = ("複製新版本失敗（robocopy 結束代碼 {0}），程式檔案可能只更新了一部分。" -f $code) +
               "請到 GitHub Releases 下載新版，解壓後覆蓋程式資料夾。"
        Set-Failed $msg -NoStart
        return $false
    }
    Remove-Item -Recurse -Force -LiteralPath (Join-Path $Root ".update") -ErrorAction SilentlyContinue
    if (-not $Packages) { Set-Content -LiteralPath $Ready -Value (Get-Date -Format o) -Encoding ascii }
    Add-Line "新版本已安裝。"
    return $true
}

$btnStart.Add_Click({
    $btnStart.Enabled = $false
    try { Start-LiveTranslate } catch { Add-Line "無法啟動：$($_.Exception.Message)" }
})

$form.Add_FormClosing({
    param($source, $e)
    $running = ($script:State -eq "steps") -and $script:Proc -and -not $script:Proc.HasExited
    if ($running) {
        $question = "套件還在更新，要中止嗎？中止後下次開啟 LiveTranslate 會重新安裝缺少的套件。"
    } elseif ($script:State -eq "wait") {
        $question = "更新還沒開始，關閉後這次不會更新，LiveTranslate 也不會自動重新開啟。要關閉嗎？"
    } else {
        return
    }
    $answer = [System.Windows.Forms.MessageBox]::Show($question, "LiveTranslate 更新",
        [System.Windows.Forms.MessageBoxButtons]::YesNo, [System.Windows.Forms.MessageBoxIcon]::Question,
        [System.Windows.Forms.MessageBoxDefaultButton]::Button2)
    if ($answer -ne [System.Windows.Forms.DialogResult]::Yes) { $e.Cancel = $true; return }
    if ($running) { Stop-StepTree }
})

# ---------- work, one step per timer tick so the window stays responsive ----------
$script:State = "wait"
$script:StepIndex = 0
$script:WaitStart = Get-Date
$script:DoneTicks = 0
Add-Line ("更新視窗：等待 LiveTranslate（pid {0}）關閉；重新安裝套件：{1}" -f $ParentPid, [bool]$Packages)

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 200
$timer.Add_Tick({
    try {
        switch ($script:State) {
            "wait" {
                $alive = ($ParentPid -gt 0) -and [bool](Get-Process -Id $ParentPid -ErrorAction SilentlyContinue)
                if ($alive) {
                    if (((Get-Date) - $script:WaitStart).TotalSeconds -gt $WaitSeconds) {
                        Set-Failed "LiveTranslate 在 $WaitSeconds 秒內沒有關閉，沒有更新套件。請先關閉它再重新更新。" -NoStart
                    }
                    return
                }
                if (-not (Test-Path -LiteralPath $PythonW)) { Set-Failed "找不到 $PythonW，請重新執行安裝。"; return }
                if ($Zip) {
                    $installed = @(Install-Release)[-1]  # the function's last output is its result
                    if ($installed -ne $true) { return }
                }
                if ($Packages) { Start-PackageSteps } else { $script:State = "launch" }
            }
            "steps" {
                if (Update-StepOutput) {
                    if (((Get-Date) - $script:StepStarted).TotalMinutes -gt $StepTimeoutMinutes) {
                        Stop-StepTree
                        Set-Failed ("{0}超過 {1} 分鐘沒有完成，已中止。" -f $Steps[$script:StepIndex].Title, $StepTimeoutMinutes)
                    }
                    return
                }
                $code = $script:Proc.ExitCode
                $script:Proc.Dispose()
                $script:Proc = $null
                if ($code -ne 0) { Set-Failed ("{0}失敗（結束代碼 {1}）。" -f $Steps[$script:StepIndex].Title, $code); return }
                $script:StepIndex++
                if ($script:StepIndex -lt $Steps.Count) { Start-Step $Steps[$script:StepIndex]; return }
                Set-Content -LiteralPath $Ready -Value (Get-Date -Format o) -Encoding ascii
                Remove-Item -Recurse -Force -LiteralPath $PipCache, $PipTemp -ErrorAction SilentlyContinue
                Add-Line "套件已更新。"
                $script:State = "launch"
            }
            "launch" {
                Start-LiveTranslate
                $script:State = "done"
            }
            "done" {
                # a LiveTranslate that ends at once (a broken package) says so here instead of nothing
                $script:DoneTicks++
                $app = $script:AppProc
                if ($app -and $app.HasExited -and $app.ExitCode -ne 0) {
                    Set-Failed ("LiveTranslate 啟動後馬上結束（結束代碼 {0}）。請用「LiveTranslate（除錯）.bat」啟動查看原因。" -f $app.ExitCode) -NoStart
                    return
                }
                if ($script:DoneTicks -ge 25) { $timer.Stop(); $form.Close() }  # 5 s
            }
        }
    } catch {
        Set-Failed ("更新時發生錯誤：" + $_.Exception.Message)
    }
})
$form.Add_Shown({ $timer.Start() })
[void]$form.ShowDialog()
$timer.Stop()
