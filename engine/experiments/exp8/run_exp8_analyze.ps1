<#
.SYNOPSIS
    Combined analysis for Experiment 8 after all 6 pairs finish.

.DESCRIPTION
    Phase 1: analyze_experiment.py --elo --plots (CSVs, generic plots,
             Bradley-Terry Elo for 4 variants).
    Phase 2: exp1_round_robin.py -- reused, since exp8 has identical 4-variant
             round-robin structure (matchup labels are exp1-compatible).

    The exp8 distinction (opening book ON + stronger parameters) is captured
    in the data itself (JSONL `from_book` flag, params in `_config.json`).
    No exp8-specific analysis script is required for the core ranking output.

    Usage:
        .\run_exp8_analyze.ps1
        .\run_exp8_analyze.ps1 -ExperimentDir C:\...\engine\out\exp8_strongest_book_20260601
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

# Resolve engine/ directory relative to this script
$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $engineDir

$AnalysisScript = Join-Path $engineDir 'analysis' 'analyze_experiment.py'
$RoundRobinScript = Join-Path $engineDir 'analysis' 'exp1_round_robin.py'

if (-not (Test-Path -LiteralPath $AnalysisScript)) {
    Write-Error "Analysis script not found: $AnalysisScript"
    exit 1
}
if (-not (Test-Path -LiteralPath $RoundRobinScript)) {
    Write-Error "Round-robin analysis script not found: $RoundRobinScript"
    exit 1
}

if (-not $ExperimentDir) {
    $latestDir = Get-ChildItem -Path 'out' -Directory |
        Where-Object { $_.Name -match '^exp8_strongest_book_' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $latestDir) {
        Write-Error "No exp8_strongest_book_* directory found in out/. Run pairs first."
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
Write-Host '  EXP 8 -- Combined Analysis (strongest variants + opening book)' -ForegroundColor Cyan
Write-Host "  Directory: $ExperimentDir" -ForegroundColor Cyan
Write-Host "  Metrics files: $metricsCount" -ForegroundColor Cyan
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host ''

# Phase 1: standard analysis
Write-Host '--- Phase 1: Standard analysis ---' -ForegroundColor Yellow
& $Python $AnalysisScript $ExperimentDir --elo --plots
$exitCode1 = $LASTEXITCODE
if ($exitCode1 -ne 0) {
    Write-Warning "Standard analysis returned $exitCode1 -- continuing with round-robin analysis"
}

Write-Host ''
# Phase 2: round-robin specific analysis (reused from exp1)
Write-Host '--- Phase 2: Round-robin analysis (reused from exp1) ---' -ForegroundColor Yellow
& $Python $RoundRobinScript $ExperimentDir
$exitCode2 = $LASTEXITCODE

if ($exitCode2 -eq 0) {
    Write-Host ''
    Write-Host '================================================================' -ForegroundColor Green
    Write-Host '  EXP 8 ANALYSIS COMPLETE' -ForegroundColor Green
    Write-Host "  Results in: $ExperimentDir" -ForegroundColor Green
    Write-Host '  Key files:' -ForegroundColor Green
    Write-Host '    analysis_wdl.csv               -- W/D/L per matchup' -ForegroundColor Green
    Write-Host '    analysis_elo.csv               -- Bradley-Terry Elo for 4 variants' -ForegroundColor Green
    Write-Host '    exp1_pair_significance.csv     -- binomial test per pair + 95% CI' -ForegroundColor Green
    Write-Host '    exp1_axis_summary.csv          -- axis A (algo) + B (eval) main effects' -ForegroundColor Green
    Write-Host '    exp1_color_advantage.csv       -- White vs Black overall' -ForegroundColor Green
    Write-Host '    exp1_round_robin_summary.txt   -- human-readable' -ForegroundColor Green
    Write-Host '    plots\exp1_pair_significance.png -- bar chart per pair with CI' -ForegroundColor Green
    Write-Host '    plots\exp1_axis_a_effect.png     -- MINIMAX vs MCTS aggregate' -ForegroundColor Green
    Write-Host '    plots\exp1_axis_b_effect.png     -- TRAD vs NN aggregate' -ForegroundColor Green
    Write-Host '    plots\exp1_wdl_matrix.png        -- 4x4 score matrix' -ForegroundColor Green
    Write-Host '================================================================' -ForegroundColor Green
}

exit $exitCode2
