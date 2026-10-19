# 翻譯模型的選擇存在 launcher\selected_model.txt（git 不追蹤）。
# 選單切換模型只改這個檔案，不改任何被 git 追蹤的腳本；否則 update.ps1 會因為
# 「本機有未提交的修改」而一直跳過自動更新。
# 用法：. (Join-Path $Root "launcher\model.ps1")，再呼叫 Get-LauncherModel / Set-LauncherModel。

$script:DefaultLauncherModel = "qwen3.5:9b"

function Test-LauncherModelName {
    # Ollama 模型標籤：英數開頭，只允許英數 . _ : / -，最長 128 字。
    # 這個值會進到 `ollama pull` 與 JSON 請求內容，所以不接受任何其他字元。
    param([string]$Name)
    return [bool]($Name -cmatch '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$')
}

function Get-LauncherModel {
    param([string]$Path)
    $default = $script:DefaultLauncherModel
    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return $default }
    try {
        $name = ([IO.File]::ReadAllText($Path)).Trim().TrimStart([char]0xFEFF).Trim()
    } catch {
        Write-Host "[注意] 讀不到 $(Split-Path -Leaf $Path)：$($_.Exception.Message)，改用預設模型 $default" -ForegroundColor Yellow
        return $default
    }
    if (-not $name) { return $default }
    if (-not (Test-LauncherModelName $name)) {
        $shown = if ($name.Length -gt 40) { $name.Substring(0, 40) + "…" } else { $name }
        Write-Host "[注意] $(Split-Path -Leaf $Path) 裡的模型名稱無效（$shown），改用預設模型 $default" -ForegroundColor Yellow
        return $default
    }
    return $name
}

function Set-LauncherModel {
    # 成功回傳 $true；名稱無效或寫入失敗回傳 $false，原本的選擇保持不變。
    param([string]$Path, [string]$Model)
    if (-not (Test-LauncherModelName $Model)) {
        Write-Host "[錯誤] 模型名稱無效：$Model" -ForegroundColor Red
        return $false
    }
    $tmp = "$Path.tmp"
    try {
        [IO.File]::WriteAllText($tmp, $Model, (New-Object Text.UTF8Encoding $false))
        Move-Item -LiteralPath $tmp -Destination $Path -Force -ErrorAction Stop
        return $true
    } catch {
        Write-Host "[錯誤] 無法寫入 $Path：$($_.Exception.Message)" -ForegroundColor Red
        if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
        return $false
    }
}
