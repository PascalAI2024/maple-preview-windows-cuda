<#
.SYNOPSIS
    Starts the full research stack: model + web search, no browser configuration.

.DESCRIPTION
    Three processes:

      :8181  mcp/search_server.py   web_search tool (Grok or MiniMax)
      :8081  llama-server           Maple-Preview on the GPU
      :8080  mcp/search_proxy.py    injects the tool, runs the tool loop

    Open http://127.0.0.1:8080 and ask something current. The proxy adds the
    search tool to every chat request server-side, so nothing needs configuring
    in the UI -- llama-server keeps MCP config in browser IndexedDB, reachable
    only through a modal, which cannot be scripted or version-controlled.

    Ctrl+C stops everything.

.EXAMPLE
    .\05-research-stack.ps1
    .\05-research-stack.ps1 -Backend minimax
#>
[CmdletBinding()]
param(
    [ValidateSet('grok', 'minimax')]
    [string]$Backend   = 'grok',
    [int]$Port         = 8080,
    [int]$UpstreamPort = 8081,
    [int]$McpPort      = 8181,
    [int]$Ctx          = 8192,
    [switch]$NoModel   # attach to an already-running llama-server
)

$ErrorActionPreference = 'Stop'
$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Definition }
$repo = Split-Path -Parent $ScriptRoot

$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { throw "python not found on PATH" }

$procs = @()

function Start-Bg {
    param([string]$File, [string[]]$Args, [string]$Label)
    Write-Host "==> $Label" -ForegroundColor Cyan
    $p = Start-Process -FilePath $File -ArgumentList $Args -PassThru -NoNewWindow
    $script:procs += $p
    return $p
}

function Wait-Http {
    param([string]$Url, [int]$Seconds = 180, [string]$Label)
    for ($i = 0; $i -lt $Seconds; $i++) {
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -eq 200) { Write-Host "    $Label ready (${i}s)" -ForegroundColor Green; return $true }
        } catch { Start-Sleep -Seconds 1 }
    }
    throw "$Label did not come up within ${Seconds}s"
}

try {
    Start-Bg -File $python -Label "search server :$McpPort (backend: $Backend)" `
             -Args @((Join-Path $repo 'mcp\search_server.py'),
                     '--port', $McpPort, '--backend', $Backend) | Out-Null
    Start-Sleep -Seconds 2

    if (-not $NoModel) {
        Start-Bg -File 'powershell.exe' -Label "llama-server :$UpstreamPort" `
                 -Args @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                         (Join-Path $ScriptRoot '04-run.ps1'),
                         '-Mode', 'server', '-Port', $UpstreamPort, '-Ctx', $Ctx) | Out-Null
        Wait-Http -Url "http://127.0.0.1:$UpstreamPort/health" -Label 'llama-server' | Out-Null
    }

    Start-Bg -File $python -Label "search proxy :$Port" `
             -Args @((Join-Path $repo 'mcp\search_proxy.py'),
                     '--listen', $Port, '--upstream', $UpstreamPort, '--mcp', $McpPort) | Out-Null
    Wait-Http -Url "http://127.0.0.1:$Port/health" -Label 'proxy' | Out-Null

    Write-Host ""
    Write-Host "  Open http://127.0.0.1:$Port  --  web search is already wired in." -ForegroundColor Green
    Write-Host "  Ctrl+C to stop." -ForegroundColor DarkGray
    Write-Host ""

    while ($true) { Start-Sleep -Seconds 3600 }
}
finally {
    Write-Host "`nstopping..." -ForegroundColor DarkGray
    foreach ($p in $procs) {
        if ($p -and -not $p.HasExited) { try { $p.Kill() } catch {} }
    }
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}
