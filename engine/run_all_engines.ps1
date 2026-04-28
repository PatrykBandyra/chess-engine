<#
.SYNOPSIS
    Runs the chess engine (main.py) in various predefined configurations for STOCKFISH, MINIMAX_TRAD, MINIMAX_NN, MCTS_TRAD, and MCTS_NN against each other.

.DESCRIPTION
    This script is a launcher for testing all major engine types against each other.
    To select a variant, UNCOMMENT EXACTLY ONE `$runArgs = @(...)` block
    in the "RUN VARIANTS" section, and leave the rest commented out.

    CLI arguments supported by main.py (short -> long name):
        -w  / --white                  : white player type (H | MINIMAX_TRAD | MINIMAX_NN | MCTS_TRAD | MCTS_NN | STOCKFISH)
        -b  / --black                  : black player type
        -m  / --mode                   : G (graphical) | B (background) | S (settings)
        -e  / --empty                  : empty board in settings mode (flag)
        -i  / --input                  : input FEN file (read from out/)
        -o  / --output                 : output FEN file (written to out/)
        -ob / --opening_book           : enable opening book (flag)
        -g  / --game                   : game moves output file (in out/)
        -l  / --logs                   : log output file (in out/)
        -dw / --depth_white            : white Minimax depth (1..20)
        -db / --depth_black            : black Minimax depth (1..20)
        -dws/ --depth_white_stockfish  : white Stockfish depth
        -dbs/ --depth_black_stockfish  : black Stockfish depth
        -sw / --skill_white            : white Stockfish skill level (0..20)
        -sb / --skill_black            : black Stockfish skill level (0..20)
        -sp / --stockfish_path         : path to Stockfish binary
        -mt / --mcts_time              : MCTS time budget in seconds
        -d  / --debug                  : debug mode (flag)

.NOTES
    The script assumes the repository layout:
        engine/main.py
        engine/out/         <- output directory (auto-created if missing)
#>

# ============================================================================
# COMMON SETTINGS
# ============================================================================
$Python = 'python'
$MainScript = 'main.py'
$RunTag = (Get-Date -Format 'yyyyMMdd_HHmmss')

# ============================================================================
# RUN VARIANTS
# Uncomment EXACTLY ONE of the `$runArgs = @(...)` blocks below.
# ============================================================================

# STOCKFISH path (edit as needed)
$StockfishPath = 'C:\\stockfish\\stockfish.exe'

# ----------------------------------------------------------------------------
# Variant 1: STOCKFISH vs MINIMAX_TRAD
# ----------------------------------------------------------------------------
$runArgs = @(
    '-w',  'STOCKFISH',
    '-b',  'MINIMAX_TRAD',
    '-m',  'B',
    '-dws','5',
    '-sw', '3',
    '-db', '4',
    '-sp', $StockfishPath,
    '-g',  "game_sf_vs_trad_$RunTag.txt",
    '-l',  "log_sf_vs_trad_$RunTag.txt"
)

# ----------------------------------------------------------------------------
# Variant 2: MINIMAX_TRAD vs STOCKFISH
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_TRAD',
#     '-b',  'STOCKFISH',
#     '-m',  'B',
#     '-dw', '4',
#     '-dbs','5',
#     '-sb', '3',
#     '-sp', $StockfishPath,
#     '-g',  "game_trad_vs_sf_$RunTag.txt",
#     '-l',  "log_trad_vs_sf_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 3: STOCKFISH vs MINIMAX_NN
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'STOCKFISH',
#     '-b',  'MINIMAX_NN',
#     '-m',  'B',
#     '-dws','5',
#     '-sw', '3',
#     '-db', '3',
#     '-sp', $StockfishPath,
#     '-g',  "game_sf_vs_nn_$RunTag.txt",
#     '-l',  "log_sf_vs_nn_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 4: MINIMAX_NN vs STOCKFISH
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_NN',
#     '-b',  'STOCKFISH',
#     '-m',  'B',
#     '-dw', '3',
#     '-dbs','5',
#     '-sb', '3',
#     '-sp', $StockfishPath,
#     '-g',  "game_nn_vs_sf_$RunTag.txt",
#     '-l',  "log_nn_vs_sf_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 5: STOCKFISH vs MCTS_TRAD
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'STOCKFISH',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-dws','5',
#     '-sw', '3',
#     '-mt', '1',
#     '-sp', $StockfishPath,
#     '-g',  "game_sf_vs_mcts_$RunTag.txt",
#     '-l',  "log_sf_vs_mcts_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 6: MCTS_TRAD vs STOCKFISH
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'STOCKFISH',
#     '-m',  'B',
#     '-mt', '1',
#     '-dbs','5',
#     '-sb', '3',
#     '-sp', $StockfishPath,
#     '-g',  "game_mcts_vs_sf_$RunTag.txt",
#     '-l',  "log_mcts_vs_sf_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 7: STOCKFISH vs MCTS_NN
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'STOCKFISH',
#     '-b',  'MCTS_NN',
#     '-m',  'B',
#     '-dws','5',
#     '-sw', '3',
#     '-mt', '1',
#     '-sp', $StockfishPath,
#     '-g',  "game_sf_vs_mctsnn_$RunTag.txt",
#     '-l',  "log_sf_vs_mctsnn_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 8: MCTS_NN vs STOCKFISH
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_NN',
#     '-b',  'STOCKFISH',
#     '-m',  'B',
#     '-mt', '1',
#     '-dbs','5',
#     '-sb', '3',
#     '-sp', $StockfishPath,
#     '-g',  "game_mctsnn_vs_sf_$RunTag.txt",
#     '-l',  "log_mctsnn_vs_sf_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 9: MINIMAX_TRAD vs MINIMAX_NN
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_TRAD',
#     '-b',  'MINIMAX_NN',
#     '-m',  'B',
#     '-dw', '4',
#     '-db', '3',
#     '-g',  "game_trad_vs_nn_$RunTag.txt",
#     '-l',  "log_trad_vs_nn_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 10: MINIMAX_NN vs MINIMAX_TRAD
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_NN',
#     '-b',  'MINIMAX_TRAD',
#     '-m',  'B',
#     '-dw', '3',
#     '-db', '4',
#     '-g',  "game_nn_vs_trad_$RunTag.txt",
#     '-l',  "log_nn_vs_trad_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 11: MINIMAX_NN vs MCTS_TRAD
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_NN',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-dw', '3',
#     '-mt', '1',
#     '-g',  "game_nn_vs_mcts_$RunTag.txt",
#     '-l',  "log_nn_vs_mcts_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 12: MCTS_TRAD vs MINIMAX_NN
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MINIMAX_NN',
#     '-m',  'B',
#     '-mt', '1',
#     '-db', '3',
#     '-g',  "game_mcts_vs_nn_$RunTag.txt",
#     '-l',  "log_mcts_vs_nn_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 13: MINIMAX_TRAD vs MCTS_TRAD
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-dw', '4',
#     '-mt', '1',
#     '-g',  "game_trad_vs_mcts_$RunTag.txt",
#     '-l',  "log_trad_vs_mcts_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 14: MCTS_TRAD vs MINIMAX_TRAD
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MINIMAX_TRAD',
#     '-m',  'B',
#     '-mt', '1',
#     '-db', '4',
#     '-g',  "game_mcts_vs_trad_$RunTag.txt",
#     '-l',  "log_mcts_vs_trad_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 15: MCTS_NN vs MINIMAX_NN
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_NN',
#     '-b',  'MINIMAX_NN',
#     '-m',  'B',
#     '-mt', '1',
#     '-db', '3',
#     '-g',  "game_mctsnn_vs_nn_$RunTag.txt",
#     '-l',  "log_mctsnn_vs_nn_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 16: MINIMAX_NN vs MCTS_NN
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_NN',
#     '-b',  'MCTS_NN',
#     '-m',  'B',
#     '-dw', '3',
#     '-mt', '1',
#     '-g',  "game_nn_vs_mctsnn_$RunTag.txt",
#     '-l',  "log_nn_vs_mctsnn_$RunTag.txt"
# )

# ============================================================================
# EXECUTION
# ============================================================================
if (-not $runArgs) {
    Write-Error 'No variant selected — uncomment one of the $runArgs blocks.'
    exit 1
}

$mainPath = Join-Path -Path "." -ChildPath $MainScript
if (-not (Test-Path -LiteralPath $mainPath)) {
    Write-Error "File not found: $mainPath. Are you running this script from the 'engine' folder?"
    exit 1
}

$outDir = Join-Path -Path "." -ChildPath 'out'
if (-not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
    Write-Host "Created directory: $outDir"
}

Write-Host '------------------------------------------------------------'
Write-Host "Running: $Python $MainScript $($runArgs -join ' ')"
Write-Host "Cwd: $(Get-Location)"
Write-Host '------------------------------------------------------------'

& $Python $MainScript @runArgs
$exitCode = $LASTEXITCODE

Write-Host '------------------------------------------------------------'
Write-Host "Finished with exit code: $exitCode"
exit $exitCode

