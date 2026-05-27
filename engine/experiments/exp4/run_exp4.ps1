<#
.SYNOPSIS
    Runs Experiment 4 (evaluation function comparison) -- all 3 sub-experiments.

.DESCRIPTION
    Phase 0 (optional): prepare 200 test positions if not present
    Phase 1: 4a -- static evaluation accuracy (TRAD/NN/SF-d1 vs SF-d20)
    Phase 2: 4b -- move agreement (each variant vs SF-d20 top-3) [slow]
    Phase 3: 4c -- evaluation speed microbenchmark

    Usage:
        .\run_exp4.ps1                            # all phases
        .\run_exp4.ps1 -SkipPrep                  # use existing test_positions.fen
        .\run_exp4.ps1 -SkipMoveAgreement         # skip slow 4b
        .\run_exp4.ps1 -Limit 20                  # quick test on fewer positions
        .\run_exp4.ps1 -SpeedN 1000               # smaller speed benchmark
#>

param(
    [string]$Python = '',
    [string]$StockfishPath = '..\stockfish_ai\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe',
    [int]$Limit = 0,
    [int]$NNDepth = 10,
    [int]$GroundTruthDepth = 20,
    [int]$MinimaxDepth = 4,
    [int]$MinimaxNNDepth = 3,
    [double]$MctsTime = 1.0,
    [int]$SpeedN = 10000,
    [switch]$SkipPrep,
    [switch]$SkipAccuracy,
    [switch]$SkipMoveAgreement,
    [switch]$SkipSpeed
)

if (-not $Python) { $Python = if ($IsMacOS -or $IsLinux) { 'python3' } else { 'python' } }

$ErrorActionPreference = 'Stop'

$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $engineDir

$expDir = "$engineDir\experiments\exp4"

# Resolve Stockfish absolute path
$sfAbs = if ([System.IO.Path]::IsPathRooted($StockfishPath)) {
    $StockfishPath
} else {
    (Resolve-Path -Path $StockfishPath -ErrorAction Stop).Path
}

if (-not (Test-Path -LiteralPath $sfAbs)) {
    Write-Error "Stockfish not found: $sfAbs"
    exit 1
}

Write-Host ''
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host '  EXPERIMENT 4 -- Evaluation function comparison' -ForegroundColor Cyan
Write-Host "  Stockfish: $sfAbs" -ForegroundColor Cyan
Write-Host "  Ground truth depth: $GroundTruthDepth" -ForegroundColor Cyan
Write-Host "  NN depth: $NNDepth" -ForegroundColor Cyan
if ($Limit -gt 0) { Write-Host "  Limit: $Limit positions (testing mode)" -ForegroundColor Yellow }
Write-Host '================================================================' -ForegroundColor Cyan

$t0 = Get-Date

# Phase 0: prepare test positions
$positionsFile = "$expDir\test_positions.fen"
if (-not $SkipPrep -and -not (Test-Path -LiteralPath $positionsFile)) {
    Write-Host ''
    Write-Host '--- Phase 0: Preparing test positions ---' -ForegroundColor Yellow
    & $Python "$expDir\prepare_test_positions.py"
    if ($LASTEXITCODE -ne 0) { Write-Error "Position preparation failed"; exit 1 }
} elseif (Test-Path -LiteralPath $positionsFile) {
    Write-Host ''
    Write-Host "--- Phase 0: Using existing $($positionsFile | Split-Path -Leaf) ---" -ForegroundColor DarkGray
}

# Phase 1: accuracy
if (-not $SkipAccuracy) {
    Write-Host ''
    Write-Host '--- Phase 1: Evaluation accuracy (4a) ---' -ForegroundColor Yellow
    $args1 = @('--stockfish', $sfAbs,
               '--nn-depth', "$NNDepth",
               '--ground-truth-depth', "$GroundTruthDepth")
    if ($Limit -gt 0) { $args1 += @('--limit', "$Limit") }
    & $Python "$expDir\run_exp4a_accuracy.py" @args1
    if ($LASTEXITCODE -ne 0) { Write-Warning "4a finished with errors" }
}

# Phase 2: move agreement (slow)
if (-not $SkipMoveAgreement) {
    Write-Host ''
    Write-Host '--- Phase 2: Move agreement (4b) -- SLOW ---' -ForegroundColor Yellow
    $args2 = @('--stockfish', $sfAbs,
               '--minimax-depth', "$MinimaxDepth",
               '--minimax-nn-depth', "$MinimaxNNDepth",
               '--mcts-time', "$MctsTime",
               '--ground-truth-depth', "$GroundTruthDepth",
               '--python', $Python)
    if ($Limit -gt 0) { $args2 += @('--limit', "$Limit") }
    & $Python "$expDir\run_exp4b_move_agreement.py" @args2
    if ($LASTEXITCODE -ne 0) { Write-Warning "4b finished with errors" }
}

# Phase 3: speed benchmark
if (-not $SkipSpeed) {
    Write-Host ''
    Write-Host '--- Phase 3: Evaluation speed (4c) ---' -ForegroundColor Yellow
    $args3 = @('--stockfish', $sfAbs,
               '--n', "$SpeedN",
               '--nn-depth', "$NNDepth")
    & $Python "$expDir\run_exp4c_eval_speed.py" @args3
    if ($LASTEXITCODE -ne 0) { Write-Warning "4c finished with errors" }
}

$elapsed = ((Get-Date) - $t0).TotalMinutes

Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host "  EXPERIMENT 4 COMPLETE in $([math]::Round($elapsed, 1)) min" -ForegroundColor Green
Write-Host "  Results in: $expDir" -ForegroundColor Green
Write-Host '  Key files:' -ForegroundColor Green
Write-Host '    exp4a_evaluations.csv         -- per-position evals' -ForegroundColor Green
Write-Host '    exp4a_accuracy_summary.csv    -- Spearman/MAE/RMSE per evaluator+phase' -ForegroundColor Green
Write-Host '    exp4b_moves.csv               -- variant moves vs SF top-3' -ForegroundColor Green
Write-Host '    exp4b_move_agreement.csv      -- match rate per variant' -ForegroundColor Green
Write-Host '    exp4c_speed_summary.csv       -- eval latency stats' -ForegroundColor Green
Write-Host '    plots\                       -- scatter plots, bar charts, histograms' -ForegroundColor Green
Write-Host '================================================================' -ForegroundColor Green
