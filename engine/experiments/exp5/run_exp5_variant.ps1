<#
.SYNOPSIS
    Runs one engine variant vs all 8 Stockfish skill levels (Experiment 5).

.DESCRIPTION
    Each variant plays 20 games against each of 8 Stockfish skill levels
    (0, 3, 5, 8, 10, 13, 15, 20). One script = one variant = 8 matchups = 160 games.

    Launch 4 instances in separate terminals for parallel execution.

    Variants:
        1  MINIMAX_TRAD d=4
        2  MINIMAX_NN   d=3
        3  MCTS_TRAD    (calibrated time from exp1)
        4  MCTS_NN      (calibrated time from exp1)

    Usage:
        .\run_exp5_variant.ps1 -Variant 1
        .\run_exp5_variant.ps1 -Variant 3 -GamesPerPair 20

    For MCTS variants, MCTS time is read from experiments\exp1\_mcts_calibrated_time.txt.
    Run experiments\exp1\run_exp1_calibrate.ps1 first if not done.

    After all 4 finish, run:
        .\run_exp5_analyze.ps1
#>

param(
    [Parameter(Mandatory)]
    [ValidateRange(1, 4)]
    [int]$Variant,

    [int]$GamesPerPair = 20,
    [double]$MctsTime = 0,
    [string]$ExperimentTag = '',
    [string]$StockfishPath = '..\stockfish_ai\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe',
    [int]$StockfishDepth = 10,
    [switch]$Gui
)

$ErrorActionPreference = 'Stop'

# Resolve engine/ directory relative to this script (experiments/exp5/ -> engine/)
$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $engineDir

if (-not (Test-Path -LiteralPath 'main.py')) {
    Write-Error "main.py not found in: $engineDir"
    exit 1
}

# ============================================================================
# READ CALIBRATED MCTS TIME (only if variant uses MCTS)
# ============================================================================

$isMcts = $Variant -in @(3, 4)
if ($isMcts -and $MctsTime -le 0) {
    $calibFile = (Resolve-Path "$PSScriptRoot\..\exp1\_mcts_calibrated_time.txt" -ErrorAction SilentlyContinue)
    if ($calibFile) {
        $MctsTime = [double](Get-Content $calibFile -Raw).Trim()
        Write-Host "Using calibrated MCTS time: $MctsTime s" -ForegroundColor Cyan
    } else {
        Write-Error "MCTS calibration file not found. Run exp1\run_exp1_calibrate.ps1 first, or pass -MctsTime."
        exit 1
    }
}

# ============================================================================
# VARIANT DEFINITIONS
# ============================================================================

$skillLevels = @(0, 3, 5, 8, 10, 13, 15, 20)

$variants = @(
    @{ name='minimax_trad_d4'; type='MINIMAX_TRAD'; extraArgs=@{ depth_white=4 } },
    @{ name='minimax_nn_d3';   type='MINIMAX_NN';   extraArgs=@{ depth_white=3 } },
    @{ name='mcts_trad';       type='MCTS_TRAD';    extraArgs=@{ mcts_time_white=$MctsTime } },
    @{ name='mcts_nn';         type='MCTS_NN';      extraArgs=@{ mcts_time_white=$MctsTime } }
)

$variantDef = $variants[$Variant - 1]
$variantName = $variantDef.name

# Build matchup list: variant vs each SF skill level
$matchups = @()
foreach ($skill in $skillLevels) {
    $matchup = @{
        white = $variantDef.type
        black = 'STOCKFISH'
        label = "${variantName}_vs_stockfish_sk${skill}"
        depth_black_stockfish = $StockfishDepth
        skill_black = $skill
    }
    # Merge variant-specific args
    foreach ($key in $variantDef.extraArgs.Keys) {
        $matchup[$key] = $variantDef.extraArgs[$key]
    }
    $matchups += $matchup
}

Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host "  EXP 5 -- Variant $Variant/4: $variantName" -ForegroundColor Green
Write-Host "  Matchups: 8 (vs Stockfish skill 0/3/5/8/10/13/15/20)" -ForegroundColor Green
Write-Host "  Games per matchup: $GamesPerPair (swap colors)" -ForegroundColor Green
Write-Host "  Total games: $($GamesPerPair * 8)" -ForegroundColor Green
if ($isMcts) {
    Write-Host "  MCTS time: $MctsTime s" -ForegroundColor Green
}
Write-Host '================================================================' -ForegroundColor Green
Write-Host ''

# ============================================================================
# GENERATE CONFIG
# ============================================================================

$configPath = "experiments\exp5\_exp5_variant${Variant}.json"
$configJson = ConvertTo-Json $matchups -Depth 4
[System.IO.File]::WriteAllText(
    (Join-Path $engineDir $configPath),
    $configJson,
    [System.Text.UTF8Encoding]::new($false)
)

# ============================================================================
# SHARED OUTPUT DIRECTORY (per day)
# ============================================================================

if (-not $ExperimentTag) {
    $ExperimentTag = Get-Date -Format 'yyyyMMdd'
}
$sharedDir = "exp5_stockfish_$ExperimentTag"

# ============================================================================
# RUN
# ============================================================================

$expArgs = @{
    ConfigFile = $configPath
    GamesPerPair = $GamesPerPair
    SwapColors = $true
    Adjudicate = $true
    OpeningsFile = 'experiments\openings_eco25.fen'
    StockfishPath = $StockfishPath
    OutputSubDir = $sharedDir
}
if ($Gui) { $expArgs.Gui = $true }

& .\experiments\run_experiment.ps1 @expArgs
$exitCode = $LASTEXITCODE

Write-Host ''
if ($exitCode -eq 0) {
    Write-Host "  Variant $Variant ($variantName) COMPLETE" -ForegroundColor Green
} else {
    Write-Host "  Variant $Variant ($variantName) finished with errors (exit $exitCode)" -ForegroundColor Red
}
Write-Host "  Output: out\$sharedDir" -ForegroundColor Cyan
Write-Host ''
Write-Host '  When all 4 variants finish, run:' -ForegroundColor Yellow
Write-Host "    $PSScriptRoot\run_exp5_analyze.ps1" -ForegroundColor Yellow

exit $exitCode
