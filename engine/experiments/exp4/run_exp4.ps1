<#
.SYNOPSIS
    Runs Experiment 4 (evaluation function comparison) -- all 3 sub-experiments.

.DESCRIPTION
    Phase 0 (optional): prepare 200 test positions if not present
    Phase 1: 4a -- TRAD eval accuracy vs SF-d20 (with SF-d1 baseline; NN excluded as biased)
    Phase 2: 4b -- TRAD variants move agreement vs SF-d20 top-3 (NN variants excluded as biased)
    Phase 3: 4c -- evaluation speed microbenchmark (TRAD vs NN -- fair, pure timing)

    Usage:
        .\run_exp4.ps1                            # all phases
        .\run_exp4.ps1 -SkipPrep                  # use existing test_positions.fen
        .\run_exp4.ps1 -SkipMoveAgreement         # skip slow 4b
        .\run_exp4.ps1 -Limit 20                  # quick test on fewer positions
        .\run_exp4.ps1 -SpeedN 1000               # smaller speed benchmark
#>

param(
    [string]$Python = '',
    [string]$StockfishPath = '',
    [int]$Limit = 0,
    [int]$NNDepth = 10,
    [int]$GroundTruthDepth = 20,
    [int]$MinimaxDepth = 3,
    [double]$MctsTime = 2.61,
    [int]$SpeedN = 10000,
    [string]$ExperimentTag = '',
    [switch]$SkipPrep,
    [switch]$SkipAccuracy,
    [switch]$SkipMoveAgreement,
    [switch]$SkipSpeed
)

if (-not $Python) {
    if ($env:VIRTUAL_ENV) {
        $Python = if ($IsMacOS -or $IsLinux) { Join-Path $env:VIRTUAL_ENV 'bin/python' } else { Join-Path $env:VIRTUAL_ENV 'Scripts/python.exe' }
    } else {
        $Python = if ($IsMacOS -or $IsLinux) { 'python3' } else { 'python' }
    }
}

# Auto-detect Stockfish binary path if not provided (matches setup_macos.sh layout)
if (-not $StockfishPath) {
    if ($IsMacOS) {
        $arch = (& uname -m).Trim()
        $sfBin = if ($arch -eq 'arm64') { 'stockfish-macos-m1-apple-silicon' } else { 'stockfish-macos-x86-64-modern' }
        $StockfishPath = "../stockfish_ai/stockfish/$sfBin"
    } elseif ($IsLinux) {
        $StockfishPath = '../stockfish_ai/stockfish/stockfish-ubuntu-x86-64-avx2'
    } else {
        $StockfishPath = '..\stockfish_ai\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe'
    }
}

$ErrorActionPreference = 'Stop'

$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $engineDir

$expDir = Join-Path $engineDir 'experiments' 'exp4'

# Output directory: out/exp4_eval_<tag>/ (consistent with Exp 1/2/3)
if (-not $ExperimentTag) {
    $ExperimentTag = Get-Date -Format 'yyyyMMdd_HHmmss'
}
$outDir = Join-Path $engineDir 'out' "exp4_eval_$ExperimentTag"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

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

# Generate _config.json with experiment metadata (reproducibility)
$sfVersion = try {
    $sfOutput = "uci`nquit" | & $sfAbs 2>$null
    ($sfOutput | Select-String 'id name Stockfish' | Select-Object -First 1).ToString().Trim()
} catch { 'unknown' }
if (-not $sfVersion) { $sfVersion = 'unknown' }

$gitCommit = try {
    (& git -C $engineDir rev-parse HEAD 2>$null).Trim()
} catch { 'unknown' }
if (-not $gitCommit) { $gitCommit = 'unknown' }

$pyVersion = try {
    (& $Python --version 2>&1).Trim()
} catch { 'unknown' }

$archStr = if ($IsMacOS -or $IsLinux) {
    try { (& uname -m 2>$null).Trim() } catch { 'unknown' }
} else {
    $env:PROCESSOR_ARCHITECTURE
}

$config = [ordered]@{
    experiment = 'exp4_evaluation_comparison'
    experiment_tag = $ExperimentTag
    timestamp = (Get-Date -Format 'o')
    args = [ordered]@{
        stockfish_path = $sfAbs
        ground_truth_depth = $GroundTruthDepth
        nn_depth = $NNDepth
        minimax_depth = $MinimaxDepth
        mcts_time = $MctsTime
        speed_n = $SpeedN
        limit = $Limit
        skip_prep = $SkipPrep.IsPresent
        skip_accuracy = $SkipAccuracy.IsPresent
        skip_move_agreement = $SkipMoveAgreement.IsPresent
        skip_speed = $SkipSpeed.IsPresent
    }
    stockfish_version = $sfVersion
    git_commit = $gitCommit
    python = "$Python ($pyVersion)"
    platform = if ($IsMacOS) { 'macOS' } elseif ($IsLinux) { 'Linux' } else { 'Windows' }
    arch = $archStr
}

$configPath = Join-Path $outDir '_config.json'
$config | ConvertTo-Json -Depth 5 | Set-Content -Path $configPath -Encoding UTF8

Write-Host ''
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host '  EXPERIMENT 4 -- Evaluation function comparison' -ForegroundColor Cyan
Write-Host "  Stockfish: $sfAbs" -ForegroundColor Cyan
Write-Host "  Ground truth depth: $GroundTruthDepth" -ForegroundColor Cyan
Write-Host "  NN depth: $NNDepth" -ForegroundColor Cyan
Write-Host "  Output dir: $outDir" -ForegroundColor Cyan
if ($Limit -gt 0) { Write-Host "  Limit: $Limit positions (testing mode)" -ForegroundColor Yellow }
Write-Host '================================================================' -ForegroundColor Cyan

$t0 = Get-Date

# Phase 0: prepare test positions
$positionsFile = Join-Path $expDir 'test_positions.fen'
if (-not $SkipPrep -and -not (Test-Path -LiteralPath $positionsFile)) {
    Write-Host ''
    Write-Host '--- Phase 0: Preparing test positions ---' -ForegroundColor Yellow
    & $Python (Join-Path $expDir 'prepare_test_positions.py')
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
               '--ground-truth-depth', "$GroundTruthDepth",
               '--output-dir', $outDir)
    if ($Limit -gt 0) { $args1 += @('--limit', "$Limit") }
    & $Python (Join-Path $expDir 'run_exp4a_accuracy.py') @args1
    if ($LASTEXITCODE -ne 0) { Write-Warning "4a finished with errors" }
}

# Phase 2: move agreement (slow)
if (-not $SkipMoveAgreement) {
    Write-Host ''
    Write-Host '--- Phase 2: Move agreement (4b) -- SLOW ---' -ForegroundColor Yellow
    $args2 = @('--stockfish', $sfAbs,
               '--minimax-depth', "$MinimaxDepth",
               '--mcts-time', "$MctsTime",
               '--ground-truth-depth', "$GroundTruthDepth",
               '--python', $Python,
               '--output-dir', $outDir)
    if ($Limit -gt 0) { $args2 += @('--limit', "$Limit") }
    & $Python (Join-Path $expDir 'run_exp4b_move_agreement.py') @args2
    if ($LASTEXITCODE -ne 0) { Write-Warning "4b finished with errors" }
}

# Phase 3: speed benchmark
if (-not $SkipSpeed) {
    Write-Host ''
    Write-Host '--- Phase 3: Evaluation speed (4c) ---' -ForegroundColor Yellow
    $args3 = @('--stockfish', $sfAbs,
               '--n', "$SpeedN",
               '--nn-depth', "$NNDepth",
               '--output-dir', $outDir)
    & $Python (Join-Path $expDir 'run_exp4c_eval_speed.py') @args3
    if ($LASTEXITCODE -ne 0) { Write-Warning "4c finished with errors" }
}

$elapsed = ((Get-Date) - $t0).TotalMinutes

Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host "  EXPERIMENT 4 COMPLETE in $([math]::Round($elapsed, 1)) min" -ForegroundColor Green
Write-Host "  Results in: $outDir" -ForegroundColor Green
Write-Host '  Key files:' -ForegroundColor Green
Write-Host '    exp4a_evaluations.csv         -- per-position evals' -ForegroundColor Green
Write-Host '    exp4a_accuracy_summary.csv    -- Spearman/MAE/RMSE per evaluator+phase' -ForegroundColor Green
Write-Host '    exp4b_moves.csv               -- variant moves vs SF top-3' -ForegroundColor Green
Write-Host '    exp4b_move_agreement.csv      -- match rate per variant' -ForegroundColor Green
Write-Host '    exp4c_speed_summary.csv       -- eval latency stats' -ForegroundColor Green
Write-Host '    plots\                       -- scatter plots, bar charts, histograms' -ForegroundColor Green
Write-Host '================================================================' -ForegroundColor Green
