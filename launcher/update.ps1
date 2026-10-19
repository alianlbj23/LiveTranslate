# 自動更新檢查（每次啟動由 start_jp_subtitles.ps1 呼叫；也可單獨執行）
# 規則：沒有 git / 不是 git 目錄 / 離線 → 略過；本機有未提交修改 → 略過；
#       只做 fast-forward；requirements.txt 有變才重新安裝套件。永遠不擋啟動。
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Continue"
$Root   = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Ready  = Join-Path $Root ".venv\.livetranslate-ready"

function Write-Info { param($m) Write-Host "[資訊] $m" -ForegroundColor Cyan }
function Write-Ok   { param($m) Write-Host "[完成] $m" -ForegroundColor Green }
function Write-Warn { param($m) Write-Host "[注意] $m" -ForegroundColor Yellow }

if (-not (Test-Path (Join-Path $Root ".git"))) { exit 0 }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Write-Warn "找不到 git，略過更新檢查。"; exit 0 }

try {
    Invoke-WebRequest -Uri "https://github.com" -Method Head -UseBasicParsing -TimeoutSec 5 | Out-Null
} catch {
    Write-Info "目前離線，略過更新檢查。"
    exit 0
}

Push-Location $Root
try {
    $branch = (git rev-parse --abbrev-ref HEAD 2>$null | Out-String).Trim()
    if (-not $branch -or $branch -eq "HEAD") { Write-Warn "不在任何分支上，略過更新。"; exit 0 }

    # 慢於 1 KB/s 超過 10 秒就放棄，避免卡在網路上
    git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=10 fetch --quiet origin $branch 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Warn "無法連到 GitHub 檢查更新，繼續使用目前版本。"; exit 0 }

    $behind = 0
    try { $behind = [int]((git rev-list --count "HEAD..origin/$branch" 2>$null | Out-String).Trim()) } catch { $behind = 0 }
    if ($behind -le 0) { Write-Ok "已是最新版本（$branch）。"; exit 0 }

    # 只看已追蹤檔案的修改；未追蹤的檔案（例如使用者自己放的東西）不妨礙 fast-forward
    $dirty = @(git status --porcelain --untracked-files=no 2>$null).Count -gt 0
    if ($dirty) { Write-Warn "GitHub 上有 $behind 個新版本，但本機有未提交的修改，略過自動更新。"; exit 0 }

    $before = (git rev-parse HEAD | Out-String).Trim()
    Write-Info "發現 $behind 個新版本，更新中..."
    git pull --ff-only --quiet origin $branch 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Warn "更新失敗（git pull），繼續使用目前版本。"; exit 0 }
    $after = (git rev-parse --short HEAD | Out-String).Trim()
    $changed = @(git diff --name-only $before HEAD 2>$null)

    if (($changed -join "`n") -match 'requirements\.txt' -and (Test-Path $Python)) {
        Write-Info "套件清單有變，更新套件（可能需要幾分鐘）..."
        Remove-Item -LiteralPath $Ready -Force -ErrorAction SilentlyContinue
        & $Python -m pip install -r (Join-Path $Root "requirements.txt") --quiet
        $ok = ($LASTEXITCODE -eq 0)
        if ($ok) { & $Python -m pip install "yasbd-lib>=0.15,<1.0" --quiet; $ok = ($LASTEXITCODE -eq 0) }
        if ($ok) { & $Python -m pip check | Out-Null; $ok = ($LASTEXITCODE -eq 0) }
        if ($ok) {
            Set-Content -LiteralPath $Ready -Value (Get-Date -Format o) -Encoding ascii
            Write-Ok "套件已更新。"
        } else {
            Write-Warn "套件更新失敗，請用選單 2) 檢查環境，或查看上方錯誤。"
        }
    }
    Write-Ok "已更新到 $after。"
    if (($changed -join "`n") -match 'launcher/') { Write-Warn "啟動腳本本身有更新，下次啟動才會生效。" }
} finally {
    Pop-Location
}
exit 0
