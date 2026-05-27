<#
.SYNOPSIS
    Combined analysis for Experiment 7 (Opening book impact).

.DESCRIPTION
    Phase 1: standard analyze_experiment.py (CSVs, generic plots)
    Phase 2: exp7_book_impact.py -- book-specific analysis:
             Per-condition W/D/L, book exit move number, opening phase time,
             chi-square and McNemar paired tests.

    Usage:
        .\run_exp7_analyze.ps1
        .\run_exp7_analyze.ps1 -ExperimentDir C:\...\engine\out\exp7_opening_book_20260527
#>

param(
    [string]$ExperimentDir = '',
    [string]$Python = 'python'
)

$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $engineDir

$AnalysisScript = Join-Path $engineDir 'analysis\analyze_experiment.py'
$Exp7Script     = Join-Path $engineDir 'analysis\exp7_book_impact.py'

if (-not (Test-Path -LiteralPath $AnalysisScript)) {
    Write-Error "Analysis script not found: $AnalysisScript"
    exit 1
}
if (-not (Test-Path -LiteralPath $Exp7Script)) {
    Write-Error "Exp7 analysis script not found: $Exp7Script"
    exit 1
}

if (-not $ExperimentDir) {
    $latestDir = Get-ChildItem -Path 'out' -Directory |
        Where-Object { $_.Name -match '^exp7_opening_book_' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $latestDir) {
        Write-Error "No exp7_opening_book_* directory found in out/. Run configs first."
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
Write-Host '  EXP 7 -- Combined Analysis (Opening book impact)' -ForegroundColor Cyan
Write-Host "  Directory: $ExperimentDir" -ForegroundColor Cyan
Write-Host "  Metrics files: $metricsCount" -ForegroundColor Cyan
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host ''

# Phase 1: standard analysis
Write-Host '--- Phase 1: Standard analysis ---' -ForegroundColor Yellow
& $Python $AnalysisScript $ExperimentDir --elo --plots
$exitCode1 = $LASTEXITCODE
if ($exitCode1 -ne 0) {
    Write-Warning "Standard analysis returned $exitCode1 -- continuing with exp7 analysis"
}

Write-Host ''
# Phase 2: exp7-specific analysis
Write-Host '--- Phase 2: Book impact analysis ---' -ForegroundColor Yellow
& $Python $Exp7Script $ExperimentDir
$exitCode2 = $LASTEXITCODE

if ($exitCode2 -eq 0) {
    Write-Host ''
    Write-Host '================================================================' -ForegroundColor Green
    Write-Host '  EXP 7 ANALYSIS COMPLETE' -ForegroundColor Green
    Write-Host "  Results in: $ExperimentDir" -ForegroundColor Green
    Write-Host '  Key files:' -ForegroundColor Green
    Write-Host '    analysis_wdl.csv                  -- raw W/D/L per matchup' -ForegroundColor Green
    Write-Host '    exp7_raw_per_game.csv             -- per-game book metrics' -ForegroundColor Green
    Write-Host '    exp7_summary.csv                  -- per-condition aggregates' -ForegroundColor Green
    Write-Host '    exp7_statistical_tests.csv        -- chi-square + McNemar' -ForegroundColor Green
    Write-Host '    exp7_summary.txt                  -- human-readable' -ForegroundColor Green
    Write-Host '    plots\exp7_wdl_comparison.png     -- book OFF vs ON' -ForegroundColor Green
    Write-Host '    plots\exp7_book_exit_hist.png     -- book exit move distribution' -ForegroundColor Green
    Write-Host '    plots\exp7_opening_time.png       -- opening phase time' -ForegroundColor Green
    Write-Host '================================================================' -ForegroundColor Green
}

exit $exitCode2
