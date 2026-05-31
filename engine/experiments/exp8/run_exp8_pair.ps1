<#
.SYNOPSIS
    Runs one matchup pair for Experiment 8 (Round-robin of strongest practical
    variants with opening book ON).

.DESCRIPTION
    Launch one instance per pair in separate terminals for parallel execution.
    All pairs write to the same shared output directory so the final analysis
    sees all 60 games together.

    Differences vs exp1:
        - Strongest practical parameters (MINIMAX_TRAD d=5, MINIMAX_NN d=4, MCTS 60s)
        - Opening book ON for both players (every pair)
        - Games start from STANDARD position (no -OpeningsFile) -- otherwise
          the book would be bypassed
        - Default 10 games per pair (5 + 5 swap) instead of 30

    Usage:
        .\run_exp8_pair.ps1 -Pair 1
        .\run_exp8_pair.ps1 -Pair 2 -ExperimentTag exp8_2
        ...
        .\run_exp8_pair.ps1 -Pair 6

    Pairs:
        1  MINIMAX_TRAD d=5 vs MINIMAX_NN d=4   (Axis B at max practical params)
        2  MINIMAX_TRAD d=5 vs MCTS_TRAD 60s    (Axis A at TRAD)
        3  MINIMAX_TRAD d=5 vs MCTS_NN 60s      (cross-axis)
        4  MINIMAX_NN d=4   vs MCTS_TRAD 60s    (cross-axis)
        5  MINIMAX_NN d=4   vs MCTS_NN 60s      (Axis A at NN)
        6  MCTS_TRAD 60s    vs MCTS_NN 60s      (Axis B at MCTS)

    After all 6 finish, run:
        .\run_exp8_analyze.ps1
#>

param(
    [Parameter(Mandatory)]
    [ValidateRange(1, 6)]
    [int]$Pair,

    [int]$GamesPerPair = 10,
    [int]$MinimaxDepthTrad = 5,
    [int]$MinimaxDepthNN = 4,
    [double]$MctsTime = 60.0,
    [string]$ExperimentTag = '',
    [string]$StockfishPath = '',
    [switch]$Gui
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

$ErrorActionPreference = 'Stop'

# Resolve engine/ directory relative to this script (experiments/exp8/ -> engine/)
$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $engineDir

if (-not (Test-Path -LiteralPath 'main.py')) {
    Write-Error "main.py not found in: $engineDir"
    exit 1
}

# ============================================================================
# PAIR DEFINITIONS -- labels match exp1 convention so exp1_round_robin.py
# can be reused for analysis (parses "<white>_vs_<black>" with VARIANTS in
# {minimax_trad, minimax_nn, mcts_trad, mcts_nn}).
# ============================================================================

$pairs = @(
    @{
        white = 'MINIMAX_TRAD'; black = 'MINIMAX_NN'
        label = 'minimax_trad_vs_minimax_nn'
        depth_white = $MinimaxDepthTrad; depth_black = $MinimaxDepthNN
    },
    @{
        white = 'MINIMAX_TRAD'; black = 'MCTS_TRAD'
        label = 'minimax_trad_vs_mcts_trad'
        depth_white = $MinimaxDepthTrad
        mcts_time_black = $MctsTime
    },
    @{
        white = 'MINIMAX_TRAD'; black = 'MCTS_NN'
        label = 'minimax_trad_vs_mcts_nn'
        depth_white = $MinimaxDepthTrad
        mcts_time_black = $MctsTime
    },
    @{
        white = 'MINIMAX_NN'; black = 'MCTS_TRAD'
        label = 'minimax_nn_vs_mcts_trad'
        depth_white = $MinimaxDepthNN
        mcts_time_black = $MctsTime
    },
    @{
        white = 'MINIMAX_NN'; black = 'MCTS_NN'
        label = 'minimax_nn_vs_mcts_nn'
        depth_white = $MinimaxDepthNN
        mcts_time_black = $MctsTime
    },
    @{
        white = 'MCTS_TRAD'; black = 'MCTS_NN'
        label = 'mcts_trad_vs_mcts_nn'
        mcts_time_white = $MctsTime; mcts_time_black = $MctsTime
    }
)

$pairDef = $pairs[$Pair - 1]
$label = $pairDef.label

Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host "  EXP 8 -- Pair $Pair/6: $($pairDef.white) vs $($pairDef.black)" -ForegroundColor Green
Write-Host "  Label: $label" -ForegroundColor Green
Write-Host "  Games: $GamesPerPair (swap colors)" -ForegroundColor Green
Write-Host "  MINIMAX_TRAD depth: $MinimaxDepthTrad" -ForegroundColor Green
Write-Host "  MINIMAX_NN   depth: $MinimaxDepthNN" -ForegroundColor Green
Write-Host "  MCTS time: $MctsTime s" -ForegroundColor Green
Write-Host "  Opening book: ON" -ForegroundColor Green
Write-Host "  Starting position: standard (no openings file)" -ForegroundColor Green
Write-Host "  Working dir: $engineDir" -ForegroundColor Green
Write-Host '================================================================' -ForegroundColor Green
Write-Host ''

# ============================================================================
# GENERATE SINGLE-PAIR CONFIG
# ============================================================================

$configPath = "experiments\exp8\_exp8_pair${Pair}.json"
$configJson = ConvertTo-Json @($pairDef) -Depth 3
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
$sharedDir = "exp8_strongest_book_$ExperimentTag"

# ============================================================================
# RUN -- book ON, no openings file (games start from move 1)
# ============================================================================

$expArgs = @{
    ConfigFile = $configPath
    GamesPerPair = $GamesPerPair
    SwapColors = $true
    Adjudicate = $true
    OpeningBook = $true
    StockfishPath = $StockfishPath
    OutputSubDir = $sharedDir
}
if ($Gui) { $expArgs.Gui = $true }

& .\experiments\run_experiment.ps1 @expArgs
$exitCode = $LASTEXITCODE

Write-Host ''
if ($exitCode -eq 0) {
    Write-Host "  Pair $Pair ($label) COMPLETE" -ForegroundColor Green
} else {
    Write-Host "  Pair $Pair ($label) finished with errors (exit $exitCode)" -ForegroundColor Red
}
Write-Host "  Output: out\$sharedDir" -ForegroundColor Cyan
Write-Host ''
Write-Host '  When all 6 pairs finish, run:' -ForegroundColor Yellow
Write-Host "    $PSScriptRoot\run_exp8_analyze.ps1" -ForegroundColor Yellow

exit $exitCode
