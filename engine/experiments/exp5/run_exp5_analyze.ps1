<#
.SYNOPSIS
    Combined analysis for Experiment 5 (Stockfish benchmark).

.DESCRIPTION
    Phase 1: standard analyze_experiment.py (CSVs, generic plots)
    Phase 2: stockfish_reval.py -- re-evaluate all games with Stockfish d=20
             (optional, can be skipped with -SkipReval; very expensive)
    Phase 3: exp5_stockfish_bench.py -- variant Elo interpolation, ACPL,
             blunder rate

    Usage:
        .\run_exp5_analyze.ps1                    # full analysis (slow due to reval)
        .\run_exp5_analyze.ps1 -SkipReval         # skip Stockfish d=20 re-eval
        .\run_exp5_analyze.ps1 -RevalDepth 15     # lower depth, ~10x faster
        .\run_exp5_analyze.ps1 -RevalLimit 20     # reval only first 20 games (testing)
#>

param(
    [string]$ExperimentDir = '',
    [string]$Python = '',
    [switch]$SkipReval,
    [int]$RevalDepth = 20,
    [int]$RevalLimit = 0,
    [string]$StockfishPath = ''
)

if (-not $StockfishPath) {
    if ($IsMacOS) {
        $StockfishPath = '../stockfish_ai/stockfish/stockfish-macos-m1-apple-silicon'
    } elseif ($IsLinux) {
        $StockfishPath = '../stockfish_ai/stockfish/stockfish-ubuntu-x86-64-avx2'
    } else {
        $StockfishPath = '..\stockfish_ai\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe'
    }
}

if (-not $Python) {
    if ($env:VIRTUAL_ENV) {
        $Python = if ($IsMacOS -or $IsLinux) { Join-Path $env:VIRTUAL_ENV 'bin/python' } else { Join-Path $env:VIRTUAL_ENV 'Scripts/python.exe' }
    } else {
        $Python = if ($IsMacOS -or $IsLinux) { 'python3' } else { 'python' }
    }
}

$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $engineDir

$AnalysisScript = Join-Path $engineDir 'analysis\analyze_experiment.py'
$RevalScript    = Join-Path $engineDir 'analysis\stockfish_reval.py'
$Exp5Script     = Join-Path $engineDir 'analysis\exp5_stockfish_bench.py'

foreach ($s in @($AnalysisScript, $RevalScript, $Exp5Script)) {
    if (-not (Test-Path -LiteralPath $s)) {
        Write-Error "Script not found: $s"
        exit 1
    }
}

if (-not $ExperimentDir) {
    $latestDir = Get-ChildItem -Path 'out' -Directory |
        Where-Object { $_.Name -match '^exp5_stockfish_' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $latestDir) {
        Write-Error "No exp5_stockfish_* directory found in out/. Run variants first."
        exit 1
    }
    $ExperimentDir = $latestDir.FullName
}

if (-not (Test-Path -LiteralPath $ExperimentDir)) {
    Write-Error "Directory not found: $ExperimentDir"
    exit 1
}

$metricsCount = (Get-ChildItem -Path $ExperimentDir -Filter 'metrics_*.jsonl').Count

Write-Host ''
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host '  EXP 5 -- Combined Analysis (Stockfish benchmark)' -ForegroundColor Cyan
Write-Host "  Directory: $ExperimentDir" -ForegroundColor Cyan
Write-Host "  Metrics files: $metricsCount" -ForegroundColor Cyan
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host ''

# Phase 1: standard analysis
Write-Host '--- Phase 1: Standard analysis ---' -ForegroundColor Yellow
& $Python $AnalysisScript $ExperimentDir --elo --plots
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Error "Standard analysis failed (exit $exitCode)"
    exit $exitCode
}

# Phase 2: Stockfish re-evaluation
if (-not $SkipReval) {
    Write-Host ''
    Write-Host '--- Phase 2: Stockfish d=' $RevalDepth ' re-evaluation (this is slow) ---' -ForegroundColor Yellow
    if (-not (Test-Path -LiteralPath $StockfishPath)) {
        Write-Warning "Stockfish not found at $StockfishPath -- skipping re-evaluation"
    } else {
        $revalArgs = @($ExperimentDir, '--stockfish', $StockfishPath, '--depth', $RevalDepth)
        if ($RevalLimit -gt 0) { $revalArgs += @('--limit', $RevalLimit) }
        & $Python $RevalScript @revalArgs
    }
} else {
    Write-Host ''
    Write-Host '--- Phase 2: Skipped (--SkipReval) ---' -ForegroundColor DarkGray
}

# Phase 3: exp5-specific analysis
Write-Host ''
Write-Host '--- Phase 3: Variant Elo + ACPL analysis ---' -ForegroundColor Yellow
& $Python $Exp5Script $ExperimentDir
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host ''
    Write-Host '================================================================' -ForegroundColor Green
    Write-Host '  EXP 5 ANALYSIS COMPLETE' -ForegroundColor Green
    Write-Host "  Results in: $ExperimentDir" -ForegroundColor Green
    Write-Host '  Key files:' -ForegroundColor Green
    Write-Host '    analysis_wdl.csv               -- raw W/D/L per matchup' -ForegroundColor Green
    Write-Host '    stockfish_reval.csv            -- Stockfish ACPL per game' -ForegroundColor Green
    Write-Host '    exp5_variant_summary.csv       -- variant scores + ACPL per SF level' -ForegroundColor Green
    Write-Host '    exp5_variant_elo.csv           -- interpolated variant Elo' -ForegroundColor Green
    Write-Host '    exp5_variant_summary.txt       -- human-readable summary' -ForegroundColor Green
    Write-Host '    plots\exp5_score_curve.png     -- performance vs SF Elo' -ForegroundColor Green
    Write-Host '    plots\exp5_acpl_by_variant.png -- ACPL comparison' -ForegroundColor Green
    Write-Host '    plots\exp5_blunder_rate.png    -- blunder rate per variant' -ForegroundColor Green
    Write-Host '================================================================' -ForegroundColor Green
}

exit $exitCode
