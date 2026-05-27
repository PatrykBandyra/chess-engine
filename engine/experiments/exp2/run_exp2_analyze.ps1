<#
.SYNOPSIS
    Combined analysis for Experiment 2 (Minimax depth scaling).

.DESCRIPTION
    Phase 1: standard analyze_experiment.py (CSVs, Elo, generic plots)
    Phase 2: exp2_depth_scaling.py — depth-specific analysis:
             Elo curve per depth, EBF / time / nodes vs depth,
             pruning techniques per depth.

    Usage:
        .\run_exp2_analyze.ps1
        .\run_exp2_analyze.ps1 -ExperimentDir C:\...\engine\out\exp2_minimax_depth_20260527
#>

param(
    [string]$ExperimentDir = '',
    [string]$Python = 'python'
)

$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $engineDir

$AnalysisScript = Join-Path $engineDir 'analysis\analyze_experiment.py'
$Exp2Script     = Join-Path $engineDir 'analysis\exp2_depth_scaling.py'

if (-not (Test-Path -LiteralPath $AnalysisScript)) {
    Write-Error "Analysis script not found: $AnalysisScript"
    exit 1
}
if (-not (Test-Path -LiteralPath $Exp2Script)) {
    Write-Error "Exp2 analysis script not found: $Exp2Script"
    exit 1
}

if (-not $ExperimentDir) {
    $latestDir = Get-ChildItem -Path 'out' -Directory |
        Where-Object { $_.Name -match '^exp2_minimax_depth_' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $latestDir) {
        Write-Error "No exp2_minimax_depth_* directory found in out/. Run matchups first."
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
Write-Host '  EXP 2 — Combined Analysis (Minimax depth scaling)' -ForegroundColor Cyan
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
# Phase 2: exp2-specific analysis
Write-Host '--- Phase 2: Depth scaling analysis ---' -ForegroundColor Yellow
& $Python $Exp2Script $ExperimentDir
$exitCode2 = $LASTEXITCODE

if ($exitCode2 -eq 0) {
    Write-Host ''
    Write-Host '================================================================' -ForegroundColor Green
    Write-Host '  EXP 2 ANALYSIS COMPLETE' -ForegroundColor Green
    Write-Host "  Results in: $ExperimentDir" -ForegroundColor Green
    Write-Host '  Key files:' -ForegroundColor Green
    Write-Host '    analysis_wdl.csv             — W/D/L per matchup' -ForegroundColor Green
    Write-Host '    analysis_elo.csv             — overall Elo (mixed groups)' -ForegroundColor Green
    Write-Host '    exp2_elo_per_depth.csv       — Elo per depth, anchored to d=4' -ForegroundColor Green
    Write-Host '    exp2_depth_summary.csv       — per-(eval, depth) metrics' -ForegroundColor Green
    Write-Host '    exp2_depth_summary.txt       — human-readable summary' -ForegroundColor Green
    Write-Host '    plots\exp2_elo_curve.png     — Elo vs depth (per evaluator)' -ForegroundColor Green
    Write-Host '    plots\exp2_ebf_curve.png     — Effective Branching Factor' -ForegroundColor Green
    Write-Host '    plots\exp2_time_curve.png    — time per move (log scale)' -ForegroundColor Green
    Write-Host '    plots\exp2_nodes_curve.png   — nodes searched (log scale)' -ForegroundColor Green
    Write-Host '    plots\exp2_pruning_by_depth.png — pruning techniques' -ForegroundColor Green
    Write-Host '================================================================' -ForegroundColor Green
}

exit $exitCode2
