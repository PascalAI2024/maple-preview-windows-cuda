<#
.SYNOPSIS
    Installs the Windows CUDA build toolchain required to compile the Maple-enabled
    llama.cpp fork: MSVC, Ninja, CMake, CUDA Toolkit.

.NOTES
    Installs are chained deliberately, not parallelised: concurrent MSI installs
    contend for the Windows Installer mutex and one of them silently fails.

    CUDA 12.8 is pinned because it appears in the fork's own CI matrix
    (.github/workflows) alongside 12.4 and 13.3, making it a known-good target.
#>
[CmdletBinding()]
param(
    [string]$CudaVersion = '12.8'
)

$ErrorActionPreference = 'Stop'

function Install-Pkg {
    param([string]$Id, [string]$Label, [string]$Version, [string]$Override)

    Write-Host "==> $Label" -ForegroundColor Cyan

    # Not $args -- that is a PowerShell automatic variable.
    $wingetArgs = @('install', '--id', $Id, '-e',
                    '--accept-source-agreements', '--accept-package-agreements',
                    '--disable-interactivity')
    if ($Version)  { $wingetArgs += @('--version', $Version) }
    if ($Override) { $wingetArgs += @('--override', $Override) }

    & winget @wingetArgs
    # winget returns 0x8A15002B when the package is already present at the
    # requested version; that is a success for our purposes.
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne -1978335189) {
        throw "$Label failed (winget exit $LASTEXITCODE)"
    }
}

Install-Pkg -Id 'Microsoft.VisualStudio.2022.BuildTools' -Label 'VS2022 Build Tools (C++ workload, ~7GB)' `
            -Override '--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'

Install-Pkg -Id 'Ninja-build.Ninja' -Label 'Ninja'
Install-Pkg -Id 'Kitware.CMake'     -Label 'CMake'
Install-Pkg -Id 'Nvidia.CUDA'       -Label "CUDA Toolkit $CudaVersion (~3GB)" -Version $CudaVersion

Write-Host ""
Write-Host "Toolchain installed. Open a NEW shell before building so PATH picks up the new tools." -ForegroundColor Green
