<#
.SYNOPSIS
    Runs Maple-Preview on the locally built fork.

.DESCRIPTION
    Four modes:
      chat      interactive session in the terminal (default)
      server    OpenAI-compatible HTTP server on :8080
      complete  one-shot prompt, prints the answer and exits -- use this to
                sanity-check a fresh build
      bench     throughput measurement via llama-bench

    Note: this fork's llama-cli dropped --no-conversation ("please use
    llama-completion instead"), so one-shot generation uses a separate binary.

    -NGL 99 offloads every layer to the GPU. The tq2_0 pack is 5.45 GB, so it
    fits a 16 GB card whole with room for KV cache.

.EXAMPLE
    .\04-run.ps1 -Mode server
    .\04-run.ps1 -Mode complete
#>
[CmdletBinding()]
param(
    [ValidateSet('chat', 'server', 'complete', 'bench')]
    [string]$Mode = 'chat',

    [string]$ModelPath = '',
    [string]$ForkDir   = '',

    [int]$NGL     = 99,
    [int]$Ctx     = 8192,
    [int]$Port    = 8080,
    # Not $Host -- that is a PowerShell automatic variable.
    [string]$BindAddress = '127.0.0.1',

    [string]$Prompt = 'What is 2+2? Answer in one short sentence.',
    [int]$NPredict  = 128,

    # Enable the web UI's MCP CORS proxy, so the browser can reach MCP servers
    # (e.g. mcp/search_server.py). Localhost only -- llama.cpp documents this as
    # unsafe in untrusted environments.
    [switch]$McpProxy
)

$ErrorActionPreference = 'Stop'

$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Definition }
. (Join-Path $ScriptRoot '_common.ps1')

# The binaries link against cudart/cublas, which live in the toolkit's bin dir
# and are not copied next to the exe.
$cudaRoot = Add-CudaToPath
if (-not $ForkDir)   { $ForkDir   = Join-Path $ScriptRoot '..\fork' }
if (-not $ModelPath) { $ModelPath = Join-Path $ScriptRoot '..\models\maple-tq2_0.gguf' }

$ForkDir   = [System.IO.Path]::GetFullPath($ForkDir)
$ModelPath = [System.IO.Path]::GetFullPath($ModelPath)

if (-not (Test-Path $ModelPath)) { throw "Model not found: $ModelPath -- run 03-download-model.ps1 first." }

$binDir     = Join-Path $ForkDir 'build\bin'
$cli        = Join-Path $binDir 'llama-cli.exe'
$server     = Join-Path $binDir 'llama-server.exe'
$completion = Join-Path $binDir 'llama-completion.exe'
$bench      = Join-Path $binDir 'llama-bench.exe'
if (-not (Test-Path $cli)) { throw "llama-cli.exe not found in $binDir -- run 02-build.ps1 first." }

Write-Host "model  $ModelPath" -ForegroundColor Cyan
Write-Host "cuda   $cudaRoot" -ForegroundColor Cyan
Write-Host "mode   $Mode  (ngl=$NGL, ctx=$Ctx)" -ForegroundColor Cyan
Write-Host ""

switch ($Mode) {
    'server' {
        # --jinja uses the chat template embedded in the GGUF (ChatML-style
        # <|im_start|>/<|im_end|>), which the tool-calling format depends on.
        $serverArgs = @('-m', $ModelPath, '-ngl', $NGL, '-c', $Ctx,
                        '--host', $BindAddress, '--port', $Port, '--jinja')
        if ($McpProxy) { $serverArgs += '--ui-mcp-proxy' }
        & $server @serverArgs
    }
    'chat' {
        & $cli -m $ModelPath -ngl $NGL -c $Ctx --jinja -cnv
    }
    'complete' {
        & $completion -m $ModelPath -ngl $NGL -c $Ctx --jinja `
                      -p $Prompt -n $NPredict
    }
    'bench' {
        # -p prompt-processing tokens, -n generation tokens.
        & $bench -m $ModelPath -ngl $NGL -p 512 -n 128
    }
}

if ($LASTEXITCODE -ne 0) { throw "llama exited with code $LASTEXITCODE" }
