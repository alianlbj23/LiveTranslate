# LiveTranslate - release builder
# Produces a zip whose only entry is LiveTranslate.exe: the first launch installs everything in
# a window (launcher\setup.ps1, run hidden: Python, Git, Ollama, packages, speech model, defaults
# for the graphics card); later launches start LiveTranslate without any console window.

param([string]$Version = "")

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

function Write-Step { param($msg) Write-Host "`n[BUILD] $msg" -ForegroundColor Cyan }
function Write-Ok   { param($msg) Write-Host "  OK: $msg" -ForegroundColor Green }

$OutDir = Join-Path $ProjectDir "release"
$Stage  = Join-Path $OutDir "LiveTranslate"

$Sha   = (& git rev-parse --short HEAD).Trim()
$Stamp = Get-Date -Format "yyyyMMdd"
$Tag   = if ($Version) { $Version } else { "$Stamp-$Sha" }
$ZipPath = Join-Path $OutDir "LiveTranslate-portable-$Tag.zip"

# Development-only files: the debug entry opens a console, the rest is not needed to run
$DropList = @("LiveTranslate（除錯）.bat", "build_release.ps1", "CLAUDE.md", "test_audio.py", ".gitignore", "screenshot")

# ── 1. Clean staging ──
Write-Step "Preparing staging directory..."
if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

# ── 2. Export tracked source via git archive ──
Write-Step "Exporting source (git archive HEAD)..."
# zip, not tar: Windows' tar.exe cannot create the non-ASCII file names of a git tar
# (快速開始.md was silently missing from the zip); Expand-Archive reads git's UTF-8 zip names
$SrcZip = Join-Path $OutDir "src.zip"
& git archive --format=zip -o $SrcZip HEAD
if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
Expand-Archive -Path $SrcZip -DestinationPath $Stage
Remove-Item $SrcZip
$tracked = @(& git ls-tree -r --name-only HEAD).Count
$exported = @(Get-ChildItem -LiteralPath $Stage -Recurse -File -Force).Count
if ($exported -ne $tracked) { throw "source export is incomplete: $exported of $tracked files" }
foreach ($d in $DropList) {
    $p = Join-Path $Stage $d
    if (Test-Path -LiteralPath $p) { Remove-Item -Recurse -Force -LiteralPath $p }
}
Write-Ok "Source exported"

# ── 3. LiveTranslate.exe (Windows' own C# compiler) ──
Write-Step "Building LiveTranslate.exe..."
& (Join-Path $ProjectDir "launcher\build_exe.ps1") -Out (Join-Path $Stage "LiveTranslate.exe")
if ($LASTEXITCODE -ne 0) { throw "LiveTranslate.exe build failed" }
Write-Ok "LiveTranslate.exe"

# ── 4. Zip ──
Write-Step "Creating archive..."
Compress-Archive -Path $Stage -DestinationPath $ZipPath -Force
$sizeMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Ok "Created $ZipPath ($sizeMb MB)"
Write-Host "`nDone. Distribute: $ZipPath" -ForegroundColor Green
