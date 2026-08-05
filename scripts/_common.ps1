<#
    Shared helpers, dot-sourced by the numbered scripts.
#>

function Resolve-ScriptRoot {
    <#
        $PSScriptRoot is not reliably populated inside a param() default block,
        so callers resolve it in the body via this helper instead.
    #>
    param([System.Management.Automation.InvocationInfo]$Invocation)

    if ($PSScriptRoot) { return $PSScriptRoot }
    return (Split-Path -Parent $Invocation.MyCommand.Definition)
}

function Resolve-CudaRoot {
    <#
        Locates the CUDA toolkit.

        $env:CUDA_PATH is only set for shells started *after* the toolkit
        install, so a fresh install would otherwise require a reboot or a new
        terminal. Falls back to globbing the standard install root and taking
        the newest version that actually contains nvcc.
    #>
    [CmdletBinding()]
    param()

    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH 'bin\nvcc.exe'))) {
        return $env:CUDA_PATH
    }

    $base = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path $base) {
        foreach ($d in (Get-ChildItem $base -Directory | Sort-Object Name -Descending)) {
            if (Test-Path (Join-Path $d.FullName 'bin\nvcc.exe')) { return $d.FullName }
        }
    }

    throw "CUDA toolkit not found. Run 01-install-toolchain.ps1 first."
}

function Add-CudaToPath {
    <#
        The built binaries link against cudart64_*.dll / cublas64_*.dll, which
        live in the toolkit's bin directory and are NOT copied next to the exe.
        Without this the loader fails with 0xC0000135 (STATUS_DLL_NOT_FOUND),
        which surfaces as the unhelpful exit code -1073741515.
    #>
    [CmdletBinding()]
    param([string]$CudaRoot)

    if (-not $CudaRoot) { $CudaRoot = Resolve-CudaRoot }
    $cudaBin = Join-Path $CudaRoot 'bin'
    if ($env:PATH -notlike "*$cudaBin*") { $env:PATH = "$cudaBin;$env:PATH" }
    return $CudaRoot
}
