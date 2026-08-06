<#
.SYNOPSIS
    Clones and builds the Maple-enabled llama.cpp fork with CUDA.

.DESCRIPTION
    Mainline llama.cpp cannot load Maple GGUFs -- the `maple` architecture and the
    `tq2_0` ternary tensor type (GGML type 35) exist only in this fork. See
    docs/why-lm-studio-cannot-run-maple.md for the evidence.

    The compute architecture is detected from the installed GPU rather than
    hardcoded, so the build targets exactly one SASS arch and stays fast.

.EXAMPLE
    .\02-build.ps1
#>
[CmdletBinding()]
param(
    [string]$ForkUrl  = 'https://github.com/stamsam/llama.cpp.git',
    [string]$Branch   = 'prism',
    [string]$ForkDir  = '',
    [string]$CudaArch = '',
    [switch]$Clean,
    # Skip the local performance patches in patches/. See docs/moe-ternary-perf-fix.md.
    [switch]$NoPatch,
    # Also build test-backend-ops, for validating CUDA kernel changes.
    # See docs/testing-kernels.md.
    [switch]$WithTests,
    # Build into an alternate directory. Lets you compile and benchmark a kernel
    # variant while a server keeps running out of the default build/.
    [string]$BuildDir = ''
)

$ErrorActionPreference = 'Stop'

# $PSScriptRoot is not reliably populated inside a param() default block, so the
# script root is resolved here in the body instead.
$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Definition }
if (-not $ForkDir) { $ForkDir = Join-Path $ScriptRoot '..\fork' }
$ForkDir = [System.IO.Path]::GetFullPath($ForkDir)

# ---------------------------------------------------------------------------
# Locate the toolchain. A freshly-installed toolchain is not on PATH until a new
# shell is opened, so everything is resolved by absolute path instead.
# ---------------------------------------------------------------------------

$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsRoot = & $vswhere -latest -products * `
                    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                    -property installationPath
        if ($vsRoot) { $vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat' }
    }
}
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found. Run 01-install-toolchain.ps1 first." }

$cmake = 'C:\Program Files\CMake\bin\cmake.exe'
if (-not (Test-Path $cmake)) {
    $cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
    if (-not $cmake) { throw "cmake not found. Run 01-install-toolchain.ps1 first." }
}

$ninjaDir = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe'
if (-not (Test-Path (Join-Path $ninjaDir 'ninja.exe'))) {
    $ninjaCmd = Get-Command ninja -ErrorAction SilentlyContinue
    if ($ninjaCmd) { $ninjaDir = Split-Path -Parent $ninjaCmd.Source }
    else { throw "ninja not found. Run 01-install-toolchain.ps1 first." }
}

# CUDA_PATH is only set for shells started *after* the toolkit install, so fall
# back to globbing the install root and taking the newest version present.
$cudaRoot = $env:CUDA_PATH
if (-not $cudaRoot -or -not (Test-Path (Join-Path $cudaRoot 'bin\nvcc.exe'))) {
    $cudaRoot = $null
    $cudaBase = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path $cudaBase) {
        foreach ($d in (Get-ChildItem $cudaBase -Directory | Sort-Object Name -Descending)) {
            if (Test-Path (Join-Path $d.FullName 'bin\nvcc.exe')) { $cudaRoot = $d.FullName; break }
        }
    }
}
if (-not $cudaRoot) { throw "CUDA toolkit not found. Run 01-install-toolchain.ps1 first." }

# ---------------------------------------------------------------------------
# Import the MSVC environment into this session.
#
# Deliberately NOT done by generating a .bat wrapper: cmd's `^` line-continuation
# parsing breaks on LF line endings, which fails silently and produces a build
# that stops with no diagnostic. Importing the vars is both robust and lets us
# invoke cmake directly with a real argument array.
# ---------------------------------------------------------------------------

Write-Host "==> Importing MSVC environment" -ForegroundColor Cyan
$envDump = & cmd /c "call `"$vcvars`" >nul 2>&1 && set"
if ($LASTEXITCODE -ne 0) { throw "vcvars64.bat failed (exit $LASTEXITCODE)" }

foreach ($line in $envDump) {
    if ($line -match '^([^=]+)=(.*)$') {
        Set-Item -Path "Env:\$($matches[1])" -Value $matches[2] -ErrorAction SilentlyContinue
    }
}

$env:CUDA_PATH = $cudaRoot
$env:PATH = "$ninjaDir;$cudaRoot\bin;$env:PATH"

Write-Host "    MSVC     $env:VCToolsVersion"
Write-Host "    CUDA     $cudaRoot"
& nvcc --version | Select-String 'release'

# ---------------------------------------------------------------------------
# Detect compute capability (8.9 -> 89)
# ---------------------------------------------------------------------------

if (-not $CudaArch) {
    $cap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim()
    if (-not $cap) { throw "Could not read compute capability from nvidia-smi." }
    $CudaArch = $cap.Replace('.', '')
    Write-Host "    GPU      compute capability $cap -> CMAKE_CUDA_ARCHITECTURES=$CudaArch"
}

# ---------------------------------------------------------------------------
# Clone / update fork
# ---------------------------------------------------------------------------

if (-not (Test-Path $ForkDir)) {
    Write-Host "==> Cloning $ForkUrl ($Branch)" -ForegroundColor Cyan
    & git clone --depth 1 --branch $Branch $ForkUrl $ForkDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
} else {
    Write-Host "==> Fork present at $ForkDir" -ForegroundColor Cyan
}
Write-Host "    rev      $(& git -C $ForkDir log -1 --format='%h %s')"

# Sanity-check that this really is the Maple fork before spending 30 min on nvcc.
$archFile = Join-Path $ForkDir 'src\llama-arch.cpp'
if (-not (Select-String -Path $archFile -Pattern 'LLM_ARCH_MAPLE' -Quiet)) {
    throw "This checkout has no LLM_ARCH_MAPLE -- wrong repo or branch."
}
Write-Host "    verified LLM_ARCH_MAPLE present" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Apply local performance patches
#
# patches/0001 gives a ~4.9x token-generation speedup by letting batch-1 MoE
# expert matmuls use the ternary MMVQ kernel. Measured, and output verified
# unchanged -- see docs/moe-ternary-perf-fix.md.
# ---------------------------------------------------------------------------

if (-not $NoPatch) {
    $patchDir = Join-Path $ScriptRoot '..\patches'
    if (Test-Path $patchDir) {
        foreach ($p in (Get-ChildItem $patchDir -Filter '*.patch' | Sort-Object Name)) {
            # --reverse --check tells us it is already applied; skip rather than fail.
            & git -C $ForkDir apply --reverse --check $p.FullName 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "    patch already applied: $($p.Name)" -ForegroundColor DarkGray
                continue
            }
            & git -C $ForkDir apply --check $p.FullName 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "    patch does not apply cleanly, skipping: $($p.Name)"
                continue
            }
            & git -C $ForkDir apply $p.FullName
            if ($LASTEXITCODE -ne 0) { throw "failed to apply $($p.Name)" }
            Write-Host "    applied patch: $($p.Name)" -ForegroundColor Green
        }
    }
} else {
    Write-Host "    -NoPatch: building upstream fork as-is" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# Configure + build
# ---------------------------------------------------------------------------

if ($BuildDir) { $buildDir = [System.IO.Path]::GetFullPath($BuildDir) }
else           { $buildDir = Join-Path $ForkDir 'build' }
if ($Clean -and (Test-Path $buildDir)) { Remove-Item -Recurse -Force $buildDir }

# LLAMA_CURL=OFF: no libcurl on a stock Windows box, and weights are fetched by
# 03-download-model.ps1 anyway.
$configureArgs = @(
    '-B', $buildDir, '-S', $ForkDir, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DGGML_CUDA=ON',
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
    "-DCUDAToolkit_ROOT=$cudaRoot",
    '-DLLAMA_CURL=OFF',
    "-DLLAMA_BUILD_TESTS=$(if ($WithTests) { 'ON' } else { 'OFF' })",
    '-DLLAMA_BUILD_EXAMPLES=OFF',
    '-DLLAMA_BUILD_TOOLS=ON'
)

Write-Host "==> Configuring" -ForegroundColor Cyan
& $cmake @configureArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed (exit $LASTEXITCODE)" }

# llama-cli    interactive chat
# llama-server OpenAI-compatible HTTP API
# llama-completion  non-interactive one-shot; llama-cli dropped --no-conversation
# llama-bench  pp/tg throughput measurement
$targets = @('llama-cli', 'llama-server', 'llama-completion', 'llama-bench')
# test-backend-ops compares every CUDA kernel against the CPU reference and can
# also time them in isolation -- the right tool for validating a kernel change.
if ($WithTests) { $targets += 'test-backend-ops' }

Write-Host "==> Building (this takes a while -- nvcc compiles one arch, sm_$CudaArch)" -ForegroundColor Cyan
& $cmake --build $buildDir --config Release --target $targets
if ($LASTEXITCODE -ne 0) { throw "cmake build failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "BUILD_COMPLETE" -ForegroundColor Green
Get-ChildItem (Join-Path $buildDir 'bin') -Filter 'llama-*.exe' -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host ("  {0,-20} {1,8:N1} MB" -f $_.Name, ($_.Length / 1MB)) }
