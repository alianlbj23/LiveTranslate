# LiveTranslate 啟動視窗（LiveTranslate.bat → menu.ps1 會先開這個；開不了才退回文字選單）
# 判斷邏輯放在獨立函式（Get-LauncherCatalog / Test-OllamaModelDownloaded / Format-ModelChoice /
# Resolve-SwitchRequest），視窗只負責顯示與回傳使用者的選擇，方便測試。
# 用法：. (Join-Path $Root "launcher\gui.ps1")；$r = Show-LauncherWindow ...；$r.Action = start | verify | guide | switch | exit

function Get-LauncherCatalog {
    # local_models.json 的模型清單（與 local_models.py 共用）
    param([string]$Root)
    $path = Join-Path $Root "local_models.json"
    $data = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
    $rows = @()
    foreach ($m in $data.models) {
        if (-not ($m.model -cmatch '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$')) { continue }
        $rows += [pscustomobject]@{ model = [string]$m.model; label = [string]$m.label; size_gb = [double]$m.size_gb; note = [string]$m.note }
    }
    return $rows
}

function Get-OllamaModelsDir {
    if ($env:OLLAMA_MODELS) { return $env:OLLAMA_MODELS }
    return (Join-Path $env:USERPROFILE ".ollama\models")
}

function Test-OllamaModelDownloaded {
    # 看 Ollama 的 manifest 檔在不在，不需要啟動 Ollama。
    # qwen3.5:9b → manifests\registry.ollama.ai\library\qwen3.5\9b
    # hf.co/Org/Repo:TAG → manifests\hf.co\Org\Repo\TAG
    param([string]$Model, [string]$ModelsDir = (Get-OllamaModelsDir))
    if (-not ($Model -cmatch '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$')) { return $false }
    $name, $tag = $Model, "latest"
    $i = $Model.LastIndexOf(":")
    if ($i -gt 0 -and $Model.Substring($i + 1) -notmatch "/") { $name = $Model.Substring(0, $i); $tag = $Model.Substring($i + 1) }
    $parts = @($name -split "/")
    if ($parts.Count -ge 2 -and $parts[0] -match "\.") {
        $rel = @($parts[0]) + $parts[1..($parts.Count - 1)]
    } elseif ($parts.Count -eq 1) {
        $rel = @("registry.ollama.ai", "library", $parts[0])
    } else {
        $rel = @("registry.ollama.ai") + $parts
    }
    $path = Join-Path $ModelsDir "manifests"
    foreach ($p in $rel) { $path = Join-Path $path $p }
    return (Test-Path -LiteralPath (Join-Path $path $tag) -PathType Leaf)
}

function Format-ModelChoice {
    param($Entry, [bool]$Downloaded, [bool]$Current)
    $state = if ($Downloaded) { "已下載" } else { ("需下載約 {0:0.0} GB" -f [double]$Entry.size_gb) }
    $text = "{0} — {1}（{2}）" -f $Entry.label, $Entry.note, $state
    if ($Current) { $text = "★ 使用中：" + $text }
    return $text
}

function Resolve-SwitchRequest {
    # 按下「切換」後要不要切換：同一個模型不動；沒下載的先問（顯示大小），使用者同意才切換。
    # 回傳 "switch" 或 "none"；$Confirm 是 { param($text) ... 回傳 $true/$false }
    param($Entry, [string]$CurrentModel, [bool]$Downloaded, [scriptblock]$Confirm)
    if ($Entry.model -eq $CurrentModel) { return "none" }
    if ($Downloaded) { return "switch" }
    $text = ("「{0}」還沒下載，需要下載約 {1:0.0} GB（下載時黑色視窗會顯示進度）。`n`n要現在下載並切換嗎？" -f $Entry.label, [double]$Entry.size_gb)
    if (& $Confirm $text) { return "switch" }
    return "none"
}

function Show-LauncherError {
    # 按鈕動作出錯時：寫進黑色視窗並顯示看得懂的訊息（不讓 WinForms 跳出例外對話框）
    param($ErrorRecord)
    $msg = "$($ErrorRecord.Exception.Message)"
    Write-Host "[錯誤] 啟動視窗：$msg" -ForegroundColor Red
    Write-Host ($ErrorRecord.ScriptStackTrace) -ForegroundColor DarkGray
    [void][System.Windows.Forms.MessageBox]::Show("發生錯誤：$msg`n`n詳細內容在黑色視窗。", "LiveTranslate",
        [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Error)
}

function New-LauncherForm {
    # 建立視窗但不顯示；回傳 @{ Form; ModelBox; Result }，Result.Action 由按鈕設定。
    # 按鈕的處理程式不使用 GetNewClosure：它只帶走當下這一層的變數，而且會看不到
    # menu.ps1 載入的函式（2026-10-20 的「The property 'Action' cannot be found」）。
    # 需要的狀態一律放在 $form.Tag，處理程式用 $this（被按的按鈕）找回所在的視窗。
    param($Catalog, [string]$CurrentModel, [string]$ModelsDir = (Get-OllamaModelsDir))
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    $result = @{ Action = "exit"; Model = $null }
    $font = New-Object System.Drawing.Font("Microsoft JhengHei UI", 10)

    $form = New-Object System.Windows.Forms.Form
    $form.Text = "LiveTranslate"
    $form.Font = $font
    $form.AutoScaleMode = [System.Windows.Forms.AutoScaleMode]::Dpi
    $form.ClientSize = New-Object System.Drawing.Size(560, 378)
    $form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $form.MaximizeBox = $false
    $form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen

    $title = New-Object System.Windows.Forms.Label
    $title.Text = "LiveTranslate"
    $title.Font = New-Object System.Drawing.Font("Microsoft JhengHei UI", 18, [System.Drawing.FontStyle]::Bold)
    $title.Location = New-Object System.Drawing.Point(20, 14); $title.AutoSize = $true
    $form.Controls.Add($title)

    $sub = New-Object System.Windows.Forms.Label
    $sub.Text = "日文直播 → 繁體中文雙字幕"
    $sub.ForeColor = [System.Drawing.Color]::DimGray
    $sub.Location = New-Object System.Drawing.Point(22, 56); $sub.AutoSize = $true
    $form.Controls.Add($sub)

    # Every action button: its Tag is the action name returned to menu.ps1
    $onAction = {
        try {
            $f = $this.FindForm()
            $f.Tag.Result.Action = [string]$this.Tag
            $f.Close()
        } catch { Show-LauncherError $_ }
    }

    $btnStart = New-Object System.Windows.Forms.Button
    $btnStart.Name = "btnStart"; $btnStart.Tag = "start"
    $btnStart.Text = "▶  啟動字幕"
    $btnStart.Font = New-Object System.Drawing.Font("Microsoft JhengHei UI", 14, [System.Drawing.FontStyle]::Bold)
    $btnStart.Location = New-Object System.Drawing.Point(20, 88); $btnStart.Size = New-Object System.Drawing.Size(520, 60)
    $btnStart.Add_Click($onAction)
    $form.Controls.Add($btnStart)

    $grp = New-Object System.Windows.Forms.GroupBox
    $grp.Text = "翻譯模型"
    $grp.Location = New-Object System.Drawing.Point(20, 160); $grp.Size = New-Object System.Drawing.Size(520, 122)
    $form.Controls.Add($grp)

    $box = New-Object System.Windows.Forms.ComboBox
    $box.Name = "cmbModel"
    $box.DropDownStyle = [System.Windows.Forms.ComboBoxStyle]::DropDownList
    $box.Location = New-Object System.Drawing.Point(14, 28); $box.Size = New-Object System.Drawing.Size(400, 28)
    $grp.Controls.Add($box)

    $entries = @($Catalog)
    $downloaded = @{}
    $selectIndex = 0
    for ($i = 0; $i -lt $entries.Count; $i++) {
        $e = $entries[$i]
        $downloaded[$e.model] = (Test-OllamaModelDownloaded -Model $e.model -ModelsDir $ModelsDir)
        [void]$box.Items.Add((Format-ModelChoice -Entry $e -Downloaded $downloaded[$e.model] -Current ($e.model -eq $CurrentModel)))
        if ($e.model -eq $CurrentModel) { $selectIndex = $i }
    }
    if ($box.Items.Count -gt 0) { $box.SelectedIndex = $selectIndex }

    $hint = New-Object System.Windows.Forms.Label
    $hint.Location = New-Object System.Drawing.Point(14, 66); $hint.Size = New-Object System.Drawing.Size(496, 46)
    $hint.ForeColor = [System.Drawing.Color]::DimGray
    $hint.Text = "選好後按「切換」，下次啟動就會使用；字幕執行中也可以在對話記錄視窗的模型選單直接切換。"
    $grp.Controls.Add($hint)

    $btnSwitch = New-Object System.Windows.Forms.Button
    $btnSwitch.Name = "btnSwitch"
    $btnSwitch.Text = "切換"
    $btnSwitch.Location = New-Object System.Drawing.Point(424, 27); $btnSwitch.Size = New-Object System.Drawing.Size(82, 30)
    $btnSwitch.Add_Click({
        try {
            $f = $this.FindForm()
            $st = $f.Tag
            if ($st.Box.SelectedIndex -lt 0) { return }
            $e = $st.Entries[$st.Box.SelectedIndex]
            $ask = $st.Confirm  # tests replace the download question with a fixed answer
            if (-not $ask) {
                $ask = { param($text) ([System.Windows.Forms.MessageBox]::Show($f, $text, "下載翻譯模型",
                    [System.Windows.Forms.MessageBoxButtons]::YesNo, [System.Windows.Forms.MessageBoxIcon]::Question) -eq
                    [System.Windows.Forms.DialogResult]::Yes) }
            }
            $decision = Resolve-SwitchRequest -Entry $e -CurrentModel $st.Current -Downloaded $st.Downloaded[$e.model] -Confirm $ask
            if ($decision -eq "switch") {
                $st.Result.Action = "switch"; $st.Result.Model = $e.model; $f.Close()
            } elseif ($e.model -eq $st.Current) {
                $st.Hint.Text = "目前已經在使用「$($e.label)」。"
            }
        } catch { Show-LauncherError $_ }
    })
    $grp.Controls.Add($btnSwitch)

    $buttons = @{}
    foreach ($spec in @(@("btnVerify", "檢查環境", 20, "verify"), @("btnGuide", "使用說明", 200, "guide"), @("btnExit", "離開", 380, "exit"))) {
        $b = New-Object System.Windows.Forms.Button
        $b.Name = $spec[0]; $b.Text = $spec[1]; $b.Tag = $spec[3]
        $b.Location = New-Object System.Drawing.Point($spec[2], 296); $b.Size = New-Object System.Drawing.Size(160, 36)
        $b.Add_Click($onAction)
        $form.Controls.Add($b)
        $buttons[$spec[0]] = $b
    }

    $foot = New-Object System.Windows.Forms.Label
    $foot.Text = "每次啟動會自動檢查更新；關閉字幕程式就會釋放顯示卡。"
    $foot.ForeColor = [System.Drawing.Color]::Gray
    $foot.Location = New-Object System.Drawing.Point(22, 344); $foot.AutoSize = $true
    $form.Controls.Add($foot)

    $form.Tag = @{ Result = $result; Box = $box; Entries = $entries; Downloaded = $downloaded; Current = $CurrentModel; Hint = $hint; Confirm = $null }
    $form.AcceptButton = $btnStart            # Enter = 啟動
    $form.CancelButton = $buttons["btnExit"]  # Esc = 離開
    $form.Add_Shown({ $this.AcceptButton.Focus() })
    return @{ Form = $form; ModelBox = $box; Result = $result }
}

function Show-LauncherWindow {
    param([string]$Root, [string]$CurrentModel)
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.Application]::EnableVisualStyles()
    $ui = New-LauncherForm -Catalog (Get-LauncherCatalog -Root $Root) -CurrentModel $CurrentModel
    [void]$ui.Form.ShowDialog()
    $ui.Form.Dispose()
    return $ui.Result
}
