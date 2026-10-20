# 最佳組合評測選單（啟動視窗按「最佳組合評測」，或文字選單進入）
# 每個項目執行 launcher\bench_run.py 的一個動作，使用者不需要輸入任何指令。
# Python 直接用這個主控台（Start-TrackedProcess -NoNewWindow），互動問題和進度列才會正常顯示。

function Invoke-BenchAction {
    param([string]$Root, [string]$Python, [string]$Action)
    $script = Join-Path $Root "launcher\bench_run.py"
    $p = Start-TrackedProcess -FilePath $Python -ArgumentList @("-I", "`"$script`"", $Action) -WorkingDirectory $Root
    return (Wait-TrackedProcess -Process $p)
}

function Get-BenchAction {
    # 選單數字 → bench_run.py 的動作；"back" = 返回主選單，$null = 無效的選項
    param([string]$Choice)
    switch ($Choice.Trim()) {
        "1" { return "download" }
        "2" { return "start" }
        "3" { return "resume" }
        "4" { return "report" }
        "5" { return "apply" }
        "0" { return "back" }
    }
    return $null
}

function Show-BenchMenu {
    param([string]$Root, [string]$Python)
    while ($true) {
        Write-Host ""
        Write-Host "==== 最佳組合評測：找出看動畫和 VTuber 最好的辨識＋翻譯組合 ====" -ForegroundColor Magenta
        [void](Invoke-BenchAction -Root $Root -Python $Python -Action "status")
        Write-Host " 1) 下載測試資料（約 550 MB，只需要一次）"
        Write-Host " 2) 開始評測（選階段與資料集，開始前顯示預估時間與費用）"
        Write-Host " 3) 從上次中斷的地方繼續"
        Write-Host " 4) 查看報告（自動打開 report.html）"
        Write-Host " 5) 套用推薦組合（寫入前先備份原本的設定）"
        Write-Host " 0) 返回"
        $choice = Read-Host "請輸入數字"
        $action = Get-BenchAction -Choice $choice
        if ($action -eq "back") { return }
        if (-not $action) {
            Write-Host "無效的選項：$choice" -ForegroundColor Yellow
            continue
        }
        $code = Invoke-BenchAction -Root $Root -Python $Python -Action $action
        if ($null -ne $code -and $code -ne 0 -and $code -ne 130) {
            Write-Host "（評測程式結束代碼 $code，詳細原因見上面的訊息或 bench_results 裡的 bench.log）" -ForegroundColor Yellow
        }
        [void](Read-Host "按 Enter 回到評測選單")
    }
}
