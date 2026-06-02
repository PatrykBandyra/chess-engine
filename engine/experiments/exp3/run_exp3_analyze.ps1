<#
.SYNOPSIS
    Combined analysis for Experiment 3 (MCTS time scaling).

.DESCRIPTION
    Phase 1: standard analyze_experiment.py (CSVs, Elo, generic plots)
    Phase 2: exp3_time_scaling.py -- time-specific analysis:
             Elo curve vs log(time) with log-linear fit,
             throughput TRAD vs NN, tree size/depth/entropy scaling.

    Usage:
        .\run_exp3_analyze.ps1
        .\run_exp3_analyze.ps1 -ExperimentDir C:\...\engine\out\exp3_mcts_time_20260527
#>

param(
    [string]$ExperimentDir = '',
    [string]$Python = ''
)

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
$Exp3Script     = Join-Path $engineDir 'analysis\exp3_time_scaling.py'

if (-not (Test-Path -LiteralPath $AnalysisScript)) {
    Write-Error "Analysis script not found: $AnalysisScript"
    exit 1
}
if (-not (Test-Path -LiteralPath $Exp3Script)) {
    Write-Error "Exp3 analysis script not found: $Exp3Script"
    exit 1
}

if (-not $ExperimentDir) {
    $latestDir = Get-ChildItem -Path 'out' -Directory |
        Where-Object { $_.Name -match '^exp3_mcts_time_' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $latestDir) {
        Write-Error "No exp3_mcts_time_* directory found in out/. Run matchups first."
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
Write-Host '  EXP 3 -- Combined Analysis (MCTS time scaling)' -ForegroundColor Cyan
Write-Host "  Directory: $ExperimentDir" -ForegroundColor Cyan
Write-Host "  Metrics files: $metricsCount" -ForegroundColor Cyan
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host ''

# Phase 1: standard analysis
Write-Host '--- Phase 1: Standard analysis ---' -ForegroundColor Yellow
& $Python $AnalysisScript $ExperimentDir --elo --plots
$exitCode1 = $LASTEXITCODE
if ($exitCode1 -ne 0) {
    Write-Error "Standard analysis failed (exit $exitCode1)"
    exit $exitCode1
}

Write-Host ''
# Phase 2: exp3-specific analysis
Write-Host '--- Phase 2: Time scaling analysis ---' -ForegroundColor Yellow
& $Python $Exp3Script $ExperimentDir
$exitCode2 = $LASTEXITCODE

if ($exitCode2 -eq 0) {
    Write-Host ''
    Write-Host '================================================================' -ForegroundColor Green
    Write-Host '  EXP 3 ANALYSIS COMPLETE' -ForegroundColor Green
    Write-Host "  Results in: $ExperimentDir" -ForegroundColor Green
    Write-Host '  Key files:' -ForegroundColor Green
    Write-Host '    analysis_wdl.csv               -- W/D/L per matchup' -ForegroundColor Green
    Write-Host '    exp3_elo_per_time.csv          -- Elo per time budget' -ForegroundColor Green
    Write-Host '    exp3_elo_log_fit.csv           -- log-linear fit (Elo per doubling)' -ForegroundColor Green
    Write-Host '    exp3_time_summary.csv          -- per-(eval, time) MCTS metrics' -ForegroundColor Green
    Write-Host '    exp3_time_summary.txt          -- human-readable summary' -ForegroundColor Green
    Write-Host '    plots\exp3_elo_curve.png       -- Elo vs log2(time) + fit' -ForegroundColor Green
    Write-Host '    plots\exp3_throughput_curve.png -- iter/s TRAD vs NN' -ForegroundColor Green
    Write-Host '    plots\exp3_tree_size_curve.png -- nodes created vs time' -ForegroundColor Green
    Write-Host '    plots\exp3_max_depth_curve.png -- tree depth vs time' -ForegroundColor Green
    Write-Host '    plots\exp3_entropy_curve.png   -- search certainty vs time' -ForegroundColor Green
    Write-Host '================================================================' -ForegroundColor Green
}

exit $exitCode2
