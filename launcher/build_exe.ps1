# 編譯 LiveTranslate.exe（launcher\LiveTranslate.cs）：用 Windows 內建的 .NET Framework C# 編譯器，
# 不用另外安裝工具。setup.ps1（安裝完成時）與 build_release.ps1（打包時）會呼叫；也可單獨執行。
param(
    [string]$Out = (Join-Path (Split-Path -Parent $PSScriptRoot) "LiveTranslate.exe")
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path -LiteralPath $csc)) { $csc = Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe" }
if (-not (Test-Path -LiteralPath $csc)) {
    Write-Host "找不到 C# 編譯器（.NET Framework 4）：$csc"
    exit 1
}
$src  = Join-Path $PSScriptRoot "LiveTranslate.cs"
$icon = Join-Path $Root "assets\LiveTranslate.ico"
$out = & $csc /nologo /target:winexe /optimize+ /codepage:65001 "/out:$Out" "/win32icon:$icon" `
    /reference:System.Windows.Forms.dll /reference:System.Drawing.dll $src
$code = $LASTEXITCODE
if ($out) { $out | ForEach-Object { Write-Host $_ } }
if ($code -ne 0) {
    Write-Host "編譯 LiveTranslate.exe 失敗（結束代碼 $code）"
    exit $code
}
Write-Host "已產生 $Out"
exit 0
