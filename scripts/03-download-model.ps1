<#
.SYNOPSIS
    Downloads a Maple-Preview GGUF from Hugging Face.

.DESCRIPTION
    Weights are never committed to this repo -- they are pulled from
    huggingface.co/stamsam/maple-preview-gguf, which hosts GGUF conversions of
    deepgrove/maple-preview (MIT).

    Variants (sizes measured, not estimated):
      tq2_0   5.45 GB  ternary, the fork's native format -- recommended
      q4_k_m  12.33 GB uniform Q4_K_M
      f16     40.5 GB  dense reference, needs >40GB VRAM or heavy offload

.EXAMPLE
    .\03-download-model.ps1 -Variant tq2_0
#>
[CmdletBinding()]
param(
    [ValidateSet('tq2_0', 'q4_k_m', 'f16')]
    [string]$Variant = 'tq2_0',

    [string]$OutDir = ''
)

$ErrorActionPreference = 'Stop'

# $PSScriptRoot is not reliably populated inside a param() default block, so the
# script root is resolved here in the body instead.
$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Definition }
if (-not $OutDir) { $OutDir = Join-Path $ScriptRoot '..\models' }

$Repo = 'stamsam/maple-preview-gguf'
$file = "maple-$Variant.gguf"
$url  = "https://huggingface.co/$Repo/resolve/main/$file"

$OutDir = [System.IO.Path]::GetFullPath($OutDir)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$dest = Join-Path $OutDir $file

# Ask the CDN how big the file is so we can verify the download afterwards.
$head = Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing
$expected = [int64]$head.Headers['Content-Length']
# Decimal GB, to match the sizes quoted on the Hugging Face repo (PowerShell's
# 1GB literal is actually a GiB and would under-report by ~7%).
Write-Host "==> $file  ($([math]::Round($expected/1e9,2)) GB) from $Repo" -ForegroundColor Cyan

if ((Test-Path $dest) -and ((Get-Item $dest).Length -eq $expected)) {
    Write-Host "    already present and correct size -- skipping" -ForegroundColor Green
} else {
    # curl.exe ships with Windows 10+ and supports resume (-C -), which matters
    # for a multi-GB pull over a flaky link. Invoke-WebRequest buffers in memory
    # and would be a poor choice at this size.
    & curl.exe -L -C - --retry 5 --retry-delay 5 -o $dest $url
    if ($LASTEXITCODE -ne 0) { throw "download failed (curl exit $LASTEXITCODE)" }
}

# --- verify -------------------------------------------------------------------
$actual = (Get-Item $dest).Length
if ($actual -ne $expected) { throw "size mismatch: got $actual, expected $expected" }

# Stream the first 4 bytes rather than ReadAllBytes: on .NET Framework
# (PowerShell 5.1) ReadAllBytes hard-fails above 2 GB, and these files are 5-40 GB.
$magicBuf = New-Object byte[] 4
$fs = [System.IO.File]::OpenRead($dest)
try { [void]$fs.Read($magicBuf, 0, 4) } finally { $fs.Dispose() }

$magicStr = -join ($magicBuf | ForEach-Object { [char]$_ })
if ($magicStr -ne 'GGUF') { throw "not a GGUF file (magic='$magicStr')" }

Write-Host "OK  $dest" -ForegroundColor Green
Write-Host "    $actual bytes, magic GGUF"
