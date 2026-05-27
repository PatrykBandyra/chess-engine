<#
.SYNOPSIS
    Combined analysis for Experiment 1 after all 6 pairs finish.

.DESCRIPTION
    Finds the shared experiment directory, runs analyze_experiment.py
    with Elo estimation and plots.

    Usage:
        .\run_exp1_analyze.ps1
        .\run_exp1_analyze.ps1 -ExperimentDir C:\...\engine\out\exp1_round_robin_20260527
#>

param(
    [string]$ExperimentDir = '',
    [string]$Python = 'python'
)

# Resolve engine/ directory relative to this script
$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $engineDir

$AnalysisScript = Join-Path $engineDir 'analysis\analyze_experiment.py'

if (-not (Test-Path -LiteralPath $AnalysisScript)) {
    Write-Error "Analysis script not found: $AnalysisScript"
    exit 1
}

if (-not $ExperimentDir) {
    $latestDir = Get-ChildItem -Path 'out' -Directory |
        Where-Object { $_.Name -match '^exp1_round_robin_' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $latestDir) {
        Write-Error "No exp1_round_robin_* directory found in out/. Run pairs first."
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
Write-Host '  EXP 1 — Combined Analysis' -ForegroundColor Cyan
Write-Host "  Directory: $ExperimentDir" -ForegroundColor Cyan
Write-Host "  Metrics files: $metricsCount" -ForegroundColor Cyan
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host ''

& $Python $AnalysisScript $ExperimentDir --elo --plots
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host ''
    Write-Host '================================================================' -ForegroundColor Green
    Write-Host '  ANALYSIS COMPLETE' -ForegroundColor Green
    Write-Host "  Results in: $ExperimentDir" -ForegroundColor Green
    Write-Host '  Key files:' -ForegroundColor Green
    Write-Host '    analysis_wdl.csv           — W/D/L per matchup' -ForegroundColor Green
    Write-Host '    analysis_elo.csv           — Elo ratings' -ForegroundColor Green
    Write-Host '    analysis_moves.csv         — per-move metrics' -ForegroundColor Green
    Write-Host '    analysis_metrics_summary.csv — aggregated stats' -ForegroundColor Green
    Write-Host '    plots\                    — visualizations' -ForegroundColor Green
    Write-Host '================================================================' -ForegroundColor Green
}

exit $exitCode
