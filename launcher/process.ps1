# 啟動 LiveTranslate 主程式並可靠地取得它的結束代碼（start_jp_subtitles.ps1 使用）。
# 用法：. (Join-Path $Root "launcher\process.ps1")

function Start-TrackedProcess {
    param([string]$FilePath, [string[]]$ArgumentList, [string]$WorkingDirectory)
    $p = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory -NoNewWindow -PassThru
    # Windows PowerShell 5.1：Start-Process -PassThru 的 Process 物件若沒在啟動後立刻取用 Handle，
    # 程序結束後 ExitCode 會是 $null，正常關閉就會被誤判成「異常結束」。
    $null = $p.Handle
    return $p
}

function Wait-TrackedProcess {
    # 等程序結束並回傳結束代碼；真的取不到時回傳 $null（呼叫端要分開處理，不可當成失敗）。
    param([System.Diagnostics.Process]$Process)
    $Process.WaitForExit()
    try {
        return $Process.ExitCode
    } catch {
        Write-Host "[注意] 讀不到結束代碼：$($_.Exception.Message)" -ForegroundColor Yellow
        return $null
    }
}
