<#
.SYNOPSIS
    Runs one engine variant on all tactical puzzles (Experiment 6).

.DESCRIPTION
    For each puzzle in puzzles.json:
        1. Run main.py with -i <fen>, engine plays as side-to-move.
        2. Stockfish skill 0 is the (fast, weak) opponent for the 2nd move.
        3. Adjudicate quickly so the run terminates.
        4. Extract the engine's first move from the game file.
        5. Compare to puzzle's best move(s).

    Output: exp6_variant<N>_<variant_name>_<tag>.csv

    Launch 4 instances in separate terminals for parallel execution.

    Variants:
        1  MINIMAX_TRAD d=4
        2  MINIMAX_NN   d=3
        3  MCTS_TRAD    (calibrated time, fallback 1.0s)
        4  MCTS_NN      (calibrated time, fallback 1.0s)

    Usage:
        .\run_exp6_variant.ps1 -Variant 1
        .\run_exp6_variant.ps1 -Variant 3 -MctsTime 5.0
        .\run_exp6_variant.ps1 -Variant 1 -Limit 10   # quick test

    After all 4 finish, run:
        .\run_exp6_analyze.ps1
#>

param(
    [Parameter(Mandatory)]
    [ValidateRange(1, 4)]
    [int]$Variant,

    [int]$MinimaxDepth = 4,
    [int]$MinimaxNNDepth = 3,
    [double]$MctsTime = 0,
    [string]$PuzzlesFile = '',
    [int]$Limit = 0,
    [string]$ExperimentTag = '',
    [string]$StockfishPath = '..\stockfish_ai\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe',
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'

$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
$exp6Dir = $PSScriptRoot
Set-Location $engineDir

if (-not (Test-Path -LiteralPath 'main.py')) {
    Write-Error "main.py not found in: $engineDir"
    exit 1
}

# Read calibrated MCTS time for MCTS variants
$isMcts = $Variant -in @(3, 4)
if ($isMcts -and $MctsTime -le 0) {
    $calibFile = Join-Path $engineDir 'experiments\exp1\_mcts_calibrated_time.txt'
    if (Test-Path -LiteralPath $calibFile) {
        $MctsTime = [double](Get-Content $calibFile -Raw).Trim()
        Write-Host "Using calibrated MCTS time: $MctsTime s" -ForegroundColor Cyan
    } else {
        $MctsTime = 1.0
        Write-Warning "No calibration file, defaulting MCTS time to $MctsTime s"
    }
}

# Resolve Stockfish absolute path
if (-not [System.IO.Path]::IsPathRooted($StockfishPath)) {
    $StockfishPath = (Resolve-Path -Path $StockfishPath -ErrorAction Stop).Path
}

# Resolve puzzles file
if (-not $PuzzlesFile) {
    $PuzzlesFile = Join-Path $exp6Dir 'puzzles.json'
}
if (-not (Test-Path -LiteralPath $PuzzlesFile)) {
    Write-Error "Puzzles file not found: $PuzzlesFile. Run prepare_puzzles.py first."
    exit 1
}

if (-not $ExperimentTag) {
    $ExperimentTag = Get-Date -Format 'yyyyMMdd'
}

# Variant definitions
$variants = @(
    @{ name='MINIMAX_TRAD_d4'; type='MINIMAX_TRAD' },
    @{ name='MINIMAX_NN_d3';   type='MINIMAX_NN' },
    @{ name='MCTS_TRAD';       type='MCTS_TRAD' },
    @{ name='MCTS_NN';         type='MCTS_NN' }
)
$variantDef = $variants[$Variant - 1]
$variantName = $variantDef.name

Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host "  EXP 6 — Variant $Variant/4: $variantName" -ForegroundColor Green
Write-Host "  Puzzles: $PuzzlesFile" -ForegroundColor Green
if ($Limit -gt 0) { Write-Host "  Limit: $Limit (testing mode)" -ForegroundColor Yellow }
Write-Host '================================================================' -ForegroundColor Green
Write-Host ''

# Determine depth/time
if ($Variant -eq 2) {
    $mm_depth = $MinimaxNNDepth
} else {
    $mm_depth = $MinimaxDepth
}

# Determine where the Python helper will write results
$resultsFile = Join-Path $exp6Dir "exp6_variant${Variant}_${variantName}_${ExperimentTag}.csv"

# Delegate the per-puzzle loop to a Python helper for robust JSON I/O
$helperPath = Join-Path $exp6Dir '_run_variant_puzzles.py'

$pyArgs = @(
    $helperPath,
    '--puzzles', $PuzzlesFile,
    '--variant-name', $variantName,
    '--variant-type', $variantDef.type,
    '--minimax-depth', $mm_depth,
    '--mcts-time', $MctsTime,
    '--stockfish-path', $StockfishPath,
    '--engine-dir', $engineDir,
    '--python', $Python,
    '--output', $resultsFile
)
if ($Limit -gt 0) { $pyArgs += @('--limit', $Limit) }

& $Python @pyArgs
$exitCode = $LASTEXITCODE

Write-Host ''
if ($exitCode -eq 0) {
    Write-Host "  Variant $Variant ($variantName) COMPLETE" -ForegroundColor Green
} else {
    Write-Host "  Variant $Variant ($variantName) finished with errors (exit $exitCode)" -ForegroundColor Red
}
Write-Host "  Output: $resultsFile" -ForegroundColor Cyan
Write-Host ''
Write-Host '  When all 4 variants finish, run:' -ForegroundColor Yellow
Write-Host "    $exp6Dir\run_exp6_analyze.ps1" -ForegroundColor Yellow

exit $exitCode
