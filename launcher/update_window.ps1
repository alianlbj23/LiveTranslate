# 更新視窗：LiveTranslate 關閉之後，必要時重新安裝套件（requirements.txt 有變時），再重新啟動 LiveTranslate。
# 由 app_update.launch_restart 以隱藏的主控台啟動，畫面上只有這個小視窗；過程同時寫進 logs\update_*.log。
param(
    [int]$ParentPid = 0,
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    [switch]$Packages
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

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
    Start-Process -FilePath $PythonW -ArgumentList "main.py" -WorkingDirectory $Root | Out-Null
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

$btnStart.Add_Click({
    $btnStart.Enabled = $false
    try { Start-LiveTranslate } catch { Add-Line "無法啟動：$($_.Exception.Message)" }
})

$form.Add_FormClosing({
    param($source, $e)
    if ($script:State -ne "steps" -or -not $script:Proc -or $script:Proc.HasExited) { return }
    $answer = [System.Windows.Forms.MessageBox]::Show(
        "套件還在更新，要中止嗎？中止後下次開啟 LiveTranslate 會重新安裝缺少的套件。", "LiveTranslate 更新",
        [System.Windows.Forms.MessageBoxButtons]::YesNo, [System.Windows.Forms.MessageBoxIcon]::Question,
        [System.Windows.Forms.MessageBoxDefaultButton]::Button2)
    if ($answer -ne [System.Windows.Forms.DialogResult]::Yes) { $e.Cancel = $true; return }
    Stop-StepTree
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
                if ($Packages) {
                    Remove-Item -LiteralPath $Ready -Force -ErrorAction SilentlyContinue
                    $script:State = "steps"
                    Start-Step $Steps[0]
                } else {
                    $script:State = "launch"
                }
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
                $script:DoneTicks++
                if ($script:DoneTicks -ge 8) { $timer.Stop(); $form.Close() }
            }
        }
    } catch {
        Set-Failed ("更新時發生錯誤：" + $_.Exception.Message)
    }
})
$form.Add_Shown({ $timer.Start() })
[void]$form.ShowDialog()
$timer.Stop()
