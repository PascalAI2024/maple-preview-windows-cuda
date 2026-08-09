[CmdletBinding()]
param(
    [string] $ModelPath,
    [string] $OutputDir,
    [ValidateRange(3, 20)]
    [int] $Repetitions = 5,
    [ValidateRange(1, 256)]
    [int] $Threads = 8
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $repoRoot 'models\maple-tq2_0.gguf'
}
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).Path

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $OutputDir = Join-Path $repoRoot "remote-results\local-controlled-$stamp"
}
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
[IO.Directory]::CreateDirectory($OutputDir) | Out-Null

$mmqBench = Join-Path $repoRoot 'fork\build-mmq-enable\bin\llama-bench.exe'
$fallbackBench = Join-Path $repoRoot 'fork\build-p7\bin\llama-bench.exe'
$testBackendOps = Join-Path $repoRoot 'fork\build-mmq-enable\bin\test-backend-ops.exe'
foreach ($required in @($mmqBench, $fallbackBench, $testBackendOps)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required binary is missing: $required"
    }
}

$activeLlama = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessName -like 'llama-*'
})
if ($activeLlama.Count -gt 0) {
    $ids = ($activeLlama.Id | Sort-Object) -join ', '
    throw "Refusing to benchmark while llama processes are active (PIDs: $ids)."
}

$utf8NoBom = [Text.UTF8Encoding]::new($false)

function Write-Utf8Lines {
    param([string] $Path, [object[]] $Lines)
    $text = @($Lines | ForEach-Object { [string] $_ })
    [IO.File]::WriteAllLines($Path, $text, $utf8NoBom)
}

function Get-GpuSnapshot {
    param([string] $Label)
    # UUID and PCI bus ID are intentionally omitted: they add no benchmark value
    # and are machine identifiers we do not need to publish.
    $query = 'timestamp,name,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.max.sm,clocks.current.memory,clocks.max.memory,memory.used,memory.total,utilization.gpu,compute_mode'
    $line = (& nvidia-smi "--query-gpu=$query" '--format=csv,noheader,nounits') -join ''
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi metadata query failed for $Label"
    }
    [pscustomobject]@{
        label = $Label
        utc = [DateTime]::UtcNow.ToString('o')
        csv = $line
    }
}

function Invoke-LlamaBench {
    param(
        [string] $Label,
        [ValidateSet('mmq', 'fallback')]
        [string] $Variant,
        [string] $Binary,
        [bool] $Record
    )

    $stderrPath = Join-Path $OutputDir "$Label.stderr"
    $arguments = @(
        '-m', $ModelPath,
        '-ngl', '99',
        '-p', '512',
        '-n', '128',
        '-r', [string] $Repetitions,
        '-t', [string] $Threads,
        '-o', 'jsonl'
    )

    $stdout = @(& $Binary @arguments 2> $stderrPath)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "llama-bench failed for $Label with exit code $exitCode"
    }

    if (-not $Record) {
        return @()
    }

    $rows = [System.Collections.Generic.List[object]]::new()
    foreach ($line in $stdout) {
        $candidate = ([string] $line).Trim()
        if (-not $candidate.StartsWith('{')) {
            continue
        }
        $parsed = $candidate | ConvertFrom-Json
        $rows.Add([pscustomobject]@{
            schema_version = 1
            run_label = $Label
            variant = $Variant
            build_commit = [string] $parsed.build_commit
            build_number = [int] $parsed.build_number
            cpu_info = ([string] $parsed.cpu_info).Trim()
            gpu_info = [string] $parsed.gpu_info
            backends = [string] $parsed.backends
            model_filename = [IO.Path]::GetFileName($ModelPath)
            model_type = [string] $parsed.model_type
            model_size = [long] $parsed.model_size
            model_n_params = [long] $parsed.model_n_params
            n_batch = [int] $parsed.n_batch
            n_ubatch = [int] $parsed.n_ubatch
            n_threads = [int] $parsed.n_threads
            n_gpu_layers = [int] $parsed.n_gpu_layers
            flash_attn = [int] $parsed.flash_attn
            n_prompt = [int] $parsed.n_prompt
            n_gen = [int] $parsed.n_gen
            test_time = if ($parsed.test_time -is [DateTime]) {
                $parsed.test_time.ToUniversalTime().ToString('o')
            } else {
                [string] $parsed.test_time
            }
            avg_ns = [long] $parsed.avg_ns
            stddev_ns = [double] $parsed.stddev_ns
            avg_ts = [double] $parsed.avg_ts
            stddev_ts = [double] $parsed.stddev_ts
            samples_ns = @($parsed.samples_ns | ForEach-Object { [long] $_ })
            samples_ts = @($parsed.samples_ts | ForEach-Object { [double] $_ })
        })
    }

    if ($rows.Count -ne 2) {
        throw "Expected pp512 and tg128 JSON rows for $Label; found $($rows.Count)"
    }
    return @($rows)
}

function Get-SeriesSummary {
    param(
        [object[]] $Rows,
        [string] $Variant,
        [int] $PromptTokens,
        [int] $GeneratedTokens
    )
    $matching = @($Rows | Where-Object {
        $_.variant -eq $Variant -and
        $_.n_prompt -eq $PromptTokens -and
        $_.n_gen -eq $GeneratedTokens
    })
    $samples = @($matching | ForEach-Object { $_.samples_ts } | ForEach-Object { [double] $_ } | Sort-Object)
    if ($samples.Count -eq 0) {
        throw "No samples for $Variant pp=$PromptTokens tg=$GeneratedTokens"
    }
    $average = ($samples | Measure-Object -Average).Average
    $variance = ($samples | ForEach-Object { [math]::Pow($_ - $average, 2) } | Measure-Object -Sum).Sum / $samples.Count
    $middle = [int] [math]::Floor($samples.Count / 2)
    $median = if (($samples.Count % 2) -eq 0) {
        ($samples[$middle - 1] + $samples[$middle]) / 2
    } else {
        $samples[$middle]
    }
    [ordered]@{
        sample_count = $samples.Count
        mean_ts = [math]::Round($average, 6)
        median_ts = [math]::Round($median, 6)
        stddev_ts = [math]::Round([math]::Sqrt($variance), 6)
        min_ts = [math]::Round($samples[0], 6)
        max_ts = [math]::Round($samples[-1], 6)
    }
}

$startedUtc = [DateTime]::UtcNow
$gpuSnapshots = [System.Collections.Generic.List[object]]::new()
$gpuSnapshots.Add((Get-GpuSnapshot -Label 'before_warmup'))

# Prime each build before the recorded A/B/B/A sequence. Windows WDDM can leave
# the GPU in a low clock state after model loading; unrecorded warmups make that
# state transition explicit instead of silently folding it into the first run.
Invoke-LlamaBench -Label 'warmup_mmq' -Variant 'mmq' -Binary $mmqBench -Record $false | Out-Null
Invoke-LlamaBench -Label 'warmup_fallback' -Variant 'fallback' -Binary $fallbackBench -Record $false | Out-Null
$gpuSnapshots.Add((Get-GpuSnapshot -Label 'after_warmup'))

$benchRows = [System.Collections.Generic.List[object]]::new()
$sequence = @(
    @{ label = 'A1_mmq'; variant = 'mmq'; binary = $mmqBench },
    @{ label = 'B1_fallback'; variant = 'fallback'; binary = $fallbackBench },
    @{ label = 'B2_fallback'; variant = 'fallback'; binary = $fallbackBench },
    @{ label = 'A2_mmq'; variant = 'mmq'; binary = $mmqBench }
)
foreach ($run in $sequence) {
    foreach ($row in @(Invoke-LlamaBench -Label $run.label -Variant $run.variant -Binary $run.binary -Record $true)) {
        $benchRows.Add($row)
    }
    $gpuSnapshots.Add((Get-GpuSnapshot -Label "after_$($run.label)"))
}

$mulMatLog = @(& $testBackendOps test -o MUL_MAT -p tq2_0 2>&1)
$mulMatExit = $LASTEXITCODE
Write-Utf8Lines -Path (Join-Path $OutputDir 'mul_mat.log') -Lines $mulMatLog
if ($mulMatExit -ne 0 -or -not (($mulMatLog -join "`n") -match '31/31')) {
    throw "MUL_MAT correctness gate failed or did not report 31/31"
}

$mulMatIdLog = @(& $testBackendOps test -o MUL_MAT_ID -p tq2_0 2>&1)
$mulMatIdExit = $LASTEXITCODE
Write-Utf8Lines -Path (Join-Path $OutputDir 'mul_mat_id.log') -Lines $mulMatIdLog
if ($mulMatIdExit -ne 0 -or -not (($mulMatIdLog -join "`n") -match '72/72')) {
    throw "MUL_MAT_ID correctness gate failed or did not report 72/72"
}

$benchJsonLines = @($benchRows | ForEach-Object { $_ | ConvertTo-Json -Depth 8 -Compress })
Write-Utf8Lines -Path (Join-Path $OutputDir 'bench.jsonl') -Lines $benchJsonLines
Write-Utf8Lines -Path (Join-Path $OutputDir 'gpu-snapshots.jsonl') -Lines @(
    $gpuSnapshots | ForEach-Object { $_ | ConvertTo-Json -Compress }
)

$mmqPp = Get-SeriesSummary -Rows $benchRows -Variant 'mmq' -PromptTokens 512 -GeneratedTokens 0
$fallbackPp = Get-SeriesSummary -Rows $benchRows -Variant 'fallback' -PromptTokens 512 -GeneratedTokens 0
$mmqTg = Get-SeriesSummary -Rows $benchRows -Variant 'mmq' -PromptTokens 0 -GeneratedTokens 128
$fallbackTg = Get-SeriesSummary -Rows $benchRows -Variant 'fallback' -PromptTokens 0 -GeneratedTokens 128

$patchHashes = @(
    Get-ChildItem -LiteralPath (Join-Path $repoRoot 'patches') -Filter '*.patch' -File |
        Sort-Object Name |
        ForEach-Object {
            $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
            [ordered]@{ file = "patches/$($_.Name)"; sha256 = $hash.Hash.ToLowerInvariant() }
        }
)

$sourceCommit = (& git -c "safe.directory=$($repoRoot -replace '\\','/')/fork" -C (Join-Path $repoRoot 'fork') rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Unable to read the fork source commit' }
$artifactCommit = (& git -C $repoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Unable to read the artifact repository commit' }

$modelHash = Get-FileHash -Algorithm SHA256 -LiteralPath $ModelPath
$binaryHashes = [ordered]@{
    mmq = (Get-FileHash -Algorithm SHA256 -LiteralPath $mmqBench).Hash.ToLowerInvariant()
    fallback = (Get-FileHash -Algorithm SHA256 -LiteralPath $fallbackBench).Hash.ToLowerInvariant()
    test_backend_ops = (Get-FileHash -Algorithm SHA256 -LiteralPath $testBackendOps).Hash.ToLowerInvariant()
}
$nvcc = Get-ChildItem -LiteralPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -Recurse -Filter 'nvcc.exe' -File -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending |
    Select-Object -First 1
$nvccText = if ($null -ne $nvcc) { @(& $nvcc.FullName --version) -join "`n" } else { '' }
$cudaToolkitVersion = if ($nvccText -match 'release\s+([0-9.]+)') { $Matches[1] } else { 'unknown' }

$result = [ordered]@{
    schema_version = 1
    status = 'complete'
    started_utc = $startedUtc.ToString('o')
    completed_utc = [DateTime]::UtcNow.ToString('o')
    host_class = 'local-controlled'
    gpu = 'NVIDIA GeForce RTX 4080 SUPER'
    compute_capability = '8.9'
    driver_version = (($gpuSnapshots[0].csv -split ',')[2]).Trim()
    cuda_toolkit_version = $cudaToolkitVersion
    benchmark_protocol = 'warmup each build, then A/B/B/A; 5 samples per metric per recorded invocation'
    repetitions_per_invocation = $Repetitions
    threads = $Threads
    correctness = [ordered]@{
        mul_mat = '31/31'
        mul_mat_id = '72/72'
    }
    pp512 = [ordered]@{
        mmq = $mmqPp
        fallback = $fallbackPp
        speedup_mean = [math]::Round($mmqPp.mean_ts / $fallbackPp.mean_ts, 6)
    }
    tg128 = [ordered]@{
        mmq = $mmqTg
        fallback = $fallbackTg
        speedup_mean = [math]::Round($mmqTg.mean_ts / $fallbackTg.mean_ts, 6)
    }
}

$manifest = [ordered]@{
    schema_version = 1
    generated_utc = [DateTime]::UtcNow.ToString('o')
    artifact_commit = $artifactCommit
    fork_commit = $sourceCommit
    cuda_toolkit_version = $cudaToolkitVersion
    patchset = '0001-0008 (0005 reverted)'
    patches = $patchHashes
    model = [ordered]@{
        repo = 'stamsam/maple-preview-gguf'
        file = [IO.Path]::GetFileName($ModelPath)
        bytes = (Get-Item -LiteralPath $ModelPath).Length
        sha256 = $modelHash.Hash.ToLowerInvariant()
    }
    binaries = $binaryHashes
    command = 'llama-bench -m <MODEL> -ngl 99 -p 512 -n 128 -r 5 -t 8 -o jsonl'
    notes = @(
        'The warmup runs are deliberately excluded from the data.',
        'Windows WDDM clock and background desktop state can materially affect short benchmark invocations.',
        'The A/B/B/A order controls for drift without claiming laboratory-grade clock locking.'
    )
}

[IO.File]::WriteAllText((Join-Path $OutputDir 'result.json'), (($result | ConvertTo-Json -Depth 10) + "`n"), $utf8NoBom)
[IO.File]::WriteAllText((Join-Path $OutputDir 'manifest.json'), (($manifest | ConvertTo-Json -Depth 10) + "`n"), $utf8NoBom)

$result | ConvertTo-Json -Depth 10
