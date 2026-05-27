<#
.SYNOPSIS
    Runs one matchup pair for Experiment 1 (Round-Robin).

.DESCRIPTION
    Launch one instance per pair in separate terminals for parallel execution.
    All pairs write to the same shared output directory so the final analysis
    sees all 300 games together.

    Usage (after running run_exp1_calibrate.ps1):
        .\run_exp1_pair.ps1 -Pair 1
        .\run_exp1_pair.ps1 -Pair 2
        ...
        .\run_exp1_pair.ps1 -Pair 6

    Pairs:
        1  MINIMAX_TRAD vs MINIMAX_NN
        2  MCTS_TRAD vs MCTS_NN
        3  MINIMAX_TRAD vs MCTS_TRAD
        4  MINIMAX_TRAD vs MCTS_NN
        5  MINIMAX_NN vs MCTS_TRAD
        6  MINIMAX_NN vs MCTS_NN

    After all 6 finish, run:
        .\run_exp1_analyze.ps1
#>

param(
    [Parameter(Mandatory)]
    [ValidateRange(1, 6)]
    [int]$Pair,

    [int]$GamesPerPair = 30,
    [double]$MctsTime = 2.7,
    [int]$MinimaxDepth = 3,
    [int]$MinimaxDepthNN = 3,
    [string]$ExperimentTag = '',
    [string]$StockfishPath = '..\stockfish_ai\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe',
    [switch]$Gui
)

$ErrorActionPreference = 'Stop'

# Resolve engine/ directory relative to this script (experiments/exp1/ -> engine/)
$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $engineDir

if (-not (Test-Path -LiteralPath 'main.py')) {
    Write-Error "main.py not found in: $engineDir"
    exit 1
}

# ============================================================================
# READ CALIBRATED MCTS TIME
# ============================================================================

$calibFile = Join-Path $PSScriptRoot '_mcts_calibrated_time.txt'

if ($MctsTime -le 0) {
    if (Test-Path -LiteralPath $calibFile) {
        $MctsTime = [double](Get-Content $calibFile -Raw).Trim()
        Write-Host "Read calibrated MCTS time: $MctsTime s (from $calibFile)" -ForegroundColor Cyan
    } else {
        Write-Error "Calibration file not found: $calibFile. Run run_exp1_calibrate.ps1 first, or pass -MctsTime manually."
        exit 1
    }
}

# ============================================================================
# PAIR DEFINITIONS
# ============================================================================

$pairs = @(
    @{
        white = 'MINIMAX_TRAD'; black = 'MINIMAX_NN'
        label = 'minimax_trad_vs_minimax_nn'
        depth_white = $MinimaxDepth; depth_black = $MinimaxDepthNN
    },
    @{
        white = 'MCTS_TRAD'; black = 'MCTS_NN'
        label = 'mcts_trad_vs_mcts_nn'
        mcts_time_white = $MctsTime; mcts_time_black = $MctsTime
    },
    @{
        white = 'MINIMAX_TRAD'; black = 'MCTS_TRAD'
        label = 'minimax_trad_vs_mcts_trad'
        depth_white = $MinimaxDepth
        mcts_time_black = $MctsTime
    },
    @{
        white = 'MINIMAX_TRAD'; black = 'MCTS_NN'
        label = 'minimax_trad_vs_mcts_nn'
        depth_white = $MinimaxDepth
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
    }
)

$pairDef = $pairs[$Pair - 1]
$label = $pairDef.label

Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host "  EXP 1 -- Pair $Pair/6: $($pairDef.white) vs $($pairDef.black)" -ForegroundColor Green
Write-Host "  Games: $GamesPerPair (swap colors)" -ForegroundColor Green
Write-Host "  MCTS time: $MctsTime s" -ForegroundColor Green
Write-Host "  Working dir: $engineDir" -ForegroundColor Green
Write-Host '================================================================' -ForegroundColor Green
Write-Host ''

# ============================================================================
# GENERATE SINGLE-PAIR CONFIG
# ============================================================================

$configPath = "experiments\exp1\_exp1_pair${Pair}.json"
$configJson = ConvertTo-Json @($pairDef) -Depth 3
[System.IO.File]::WriteAllText(
    (Join-Path $engineDir $configPath),
    $configJson,
    [System.Text.UTF8Encoding]::new($false)
)

# ============================================================================
# SHARED OUTPUT DIRECTORY
# ============================================================================

if (-not $ExperimentTag) {
    $ExperimentTag = Get-Date -Format 'yyyyMMdd'
}
$sharedDir = "exp1_round_robin_$ExperimentTag"

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
    Write-Host "  Pair $Pair ($label) COMPLETE" -ForegroundColor Green
} else {
    Write-Host "  Pair $Pair ($label) finished with errors (exit $exitCode)" -ForegroundColor Red
}
Write-Host "  Output: out\$sharedDir" -ForegroundColor Cyan
Write-Host ''
Write-Host '  When all 6 pairs finish, run:' -ForegroundColor Yellow
Write-Host "    $PSScriptRoot\run_exp1_analyze.ps1" -ForegroundColor Yellow

exit $exitCode
