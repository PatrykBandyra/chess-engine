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
$Exp1Script     = Join-Path $engineDir 'analysis\exp1_round_robin.py'

if (-not (Test-Path -LiteralPath $AnalysisScript)) {
    Write-Error "Analysis script not found: $AnalysisScript"
    exit 1
}
if (-not (Test-Path -LiteralPath $Exp1Script)) {
    Write-Error "Exp1 analysis script not found: $Exp1Script"
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

# Phase 1: standard analysis
Write-Host '--- Phase 1: Standard analysis ---' -ForegroundColor Yellow
& $Python $AnalysisScript $ExperimentDir --elo --plots
$exitCode1 = $LASTEXITCODE
if ($exitCode1 -ne 0) {
    Write-Warning "Standard analysis returned $exitCode1 — continuing with exp1 analysis"
}

Write-Host ''
# Phase 2: exp1-specific round-robin analysis
Write-Host '--- Phase 2: Round-robin specific analysis ---' -ForegroundColor Yellow
& $Python $Exp1Script $ExperimentDir
$exitCode2 = $LASTEXITCODE

if ($exitCode2 -eq 0) {
    Write-Host ''
    Write-Host '================================================================' -ForegroundColor Green
    Write-Host '  EXP 1 ANALYSIS COMPLETE' -ForegroundColor Green
    Write-Host "  Results in: $ExperimentDir" -ForegroundColor Green
    Write-Host '  Key files:' -ForegroundColor Green
    Write-Host '    analysis_wdl.csv               — W/D/L per matchup (generic)' -ForegroundColor Green
    Write-Host '    analysis_elo.csv               — Bradley-Terry Elo (generic)' -ForegroundColor Green
    Write-Host '    exp1_pair_significance.csv     — binomial test per pair + 95% CI' -ForegroundColor Green
    Write-Host '    exp1_axis_summary.csv          — axis A (algo) + B (eval) main effects' -ForegroundColor Green
    Write-Host '    exp1_color_advantage.csv       — White vs Black overall' -ForegroundColor Green
    Write-Host '    exp1_round_robin_summary.txt   — human-readable' -ForegroundColor Green
    Write-Host '    plots\exp1_pair_significance.png — bar chart per pair with CI' -ForegroundColor Green
    Write-Host '    plots\exp1_axis_a_effect.png     — MINIMAX vs MCTS aggregate' -ForegroundColor Green
    Write-Host '    plots\exp1_axis_b_effect.png     — TRAD vs NN aggregate' -ForegroundColor Green
    Write-Host '    plots\exp1_wdl_matrix.png        — 4x4 score matrix' -ForegroundColor Green
    Write-Host '================================================================' -ForegroundColor Green
}

exit $exitCode2
