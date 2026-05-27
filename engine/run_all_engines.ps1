<#
.SYNOPSIS
    Runs the chess engine (main.py) in various predefined cross-engine
    configurations: MCTS_TRAD / MCTS_NN / MINIMAX_TRAD / MINIMAX_NN vs
    STOCKFISH, plus engine-vs-engine comparisons.

.DESCRIPTION
    This script is a launcher for testing all major engine types against
    each other. To select a variant, UNCOMMENT EXACTLY ONE
    `$runArgs = @(...)` block in the "RUN VARIANTS" section, and leave
    the rest commented out.

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
        -mtw / --mcts_time_white       : MCTS time budget in seconds for white
        -mtb / --mcts_time_black       : MCTS time budget in seconds for black
        -d  / --debug                  : debug mode (flag)

.NOTES
    Run this script from the engine/ folder.
        engine/main.py
        engine/out/         <- output directory (auto-created if missing)
#>

# ============================================================================
# COMMON SETTINGS
# ============================================================================

# Path to the Python interpreter. Auto-selects 'python3' on macOS/Linux, 'python' on Windows.
# For a specific venv/conda env, set the full path, e.g.:
# $Python = 'C:\Users\me\anaconda3\envs\chess-engine\python.exe'
$Python = if ($IsMacOS -or $IsLinux) { 'python3' } else { 'python' }

# Entry-point script
$MainScript = 'main.py'

# Common suffix for output filenames - useful for tagging experiment batches
$RunTag = (Get-Date -Format 'yyyyMMdd_HHmmss')

# Stockfish binary path (edit if your Stockfish lives elsewhere)
$StockfishPath = '..\stockfish_ai\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe'

# ============================================================================
# RUN VARIANTS
# Uncomment EXACTLY ONE of the `$runArgs = @(...)` blocks below.
# ============================================================================

# ----------------------------------------------------------------------------
# Variant 1: MCTS_TRAD vs STOCKFISH (depth 5, skill 3), 1s per side
# ----------------------------------------------------------------------------
$runArgs = @(
    '-w',  'MCTS_TRAD',
    '-b',  'STOCKFISH',
    '-m',  'B',
    '-mtw','1', '-mtb','1',
    '-dbs','5', '-sb','3',
    '-sp', $StockfishPath,
    '-g',  "game_mcts_trad_vs_sf_$RunTag.txt",
    '-l',  "log_mcts_trad_vs_sf_$RunTag.txt"
)

# ----------------------------------------------------------------------------
# Variant 2: STOCKFISH (depth 5, skill 3) vs MCTS_TRAD, 1s per side
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'STOCKFISH',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-dws','5', '-sw','3',
#     '-mtw','1', '-mtb','1',
#     '-sp', $StockfishPath,
#     '-g',  "game_sf_vs_mcts_trad_$RunTag.txt",
#     '-l',  "log_sf_vs_mcts_trad_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 3: MCTS_NN vs STOCKFISH (depth 5, skill 3), 1s per side
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_NN',
#     '-b',  'STOCKFISH',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-dbs','5', '-sb','3',
#     '-sp', $StockfishPath,
#     '-g',  "game_mcts_nn_vs_sf_$RunTag.txt",
#     '-l',  "log_mcts_nn_vs_sf_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 4: STOCKFISH (depth 5, skill 3) vs MCTS_NN, 1s per side
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'STOCKFISH',
#     '-b',  'MCTS_NN',
#     '-m',  'B',
#     '-dws','5', '-sw','3',
#     '-mtw','1', '-mtb','1',
#     '-sp', $StockfishPath,
#     '-g',  "game_sf_vs_mcts_nn_$RunTag.txt",
#     '-l',  "log_sf_vs_mcts_nn_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 5: MINIMAX_TRAD (depth 4) vs STOCKFISH (depth 5, skill 3)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_TRAD',
#     '-b',  'STOCKFISH',
#     '-m',  'B',
#     '-dw', '4',
#     '-dbs','5', '-sb','3',
#     '-sp', $StockfishPath,
#     '-g',  "game_minimax_trad_vs_sf_$RunTag.txt",
#     '-l',  "log_minimax_trad_vs_sf_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 6: STOCKFISH (depth 5, skill 3) vs MINIMAX_TRAD (depth 4)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'STOCKFISH',
#     '-b',  'MINIMAX_TRAD',
#     '-m',  'B',
#     '-dws','5', '-sw','3',
#     '-db', '4',
#     '-sp', $StockfishPath,
#     '-g',  "game_sf_vs_minimax_trad_$RunTag.txt",
#     '-l',  "log_sf_vs_minimax_trad_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 7: MINIMAX_NN (depth 3) vs STOCKFISH (depth 5, skill 3)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_NN',
#     '-b',  'STOCKFISH',
#     '-m',  'B',
#     '-dw', '3',
#     '-dbs','5', '-sb','3',
#     '-sp', $StockfishPath,
#     '-g',  "game_minimax_nn_vs_sf_$RunTag.txt",
#     '-l',  "log_minimax_nn_vs_sf_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 8: STOCKFISH (depth 5, skill 3) vs MINIMAX_NN (depth 3)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'STOCKFISH',
#     '-b',  'MINIMAX_NN',
#     '-m',  'B',
#     '-dws','5', '-sw','3',
#     '-db', '3',
#     '-sp', $StockfishPath,
#     '-g',  "game_sf_vs_minimax_nn_$RunTag.txt",
#     '-l',  "log_sf_vs_minimax_nn_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 9: MCTS_TRAD vs STOCKFISH, with opening book enabled
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'STOCKFISH',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-dbs','5', '-sb','3',
#     '-ob',
#     '-sp', $StockfishPath,
#     '-g',  "game_mcts_trad_vs_sf_ob_$RunTag.txt",
#     '-l',  "log_mcts_trad_vs_sf_ob_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 10: STOCKFISH vs MCTS_TRAD, with opening book enabled
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'STOCKFISH',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-dws','5', '-sw','3',
#     '-mtw','1', '-mtb','1',
#     '-ob',
#     '-sp', $StockfishPath,
#     '-g',  "game_sf_vs_mcts_trad_ob_$RunTag.txt",
#     '-l',  "log_sf_vs_mcts_trad_ob_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 11: MCTS_NN vs STOCKFISH, with opening book enabled
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_NN',
#     '-b',  'STOCKFISH',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-dbs','5', '-sb','3',
#     '-ob',
#     '-sp', $StockfishPath,
#     '-g',  "game_mcts_nn_vs_sf_ob_$RunTag.txt",
#     '-l',  "log_mcts_nn_vs_sf_ob_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 12: MINIMAX_TRAD vs STOCKFISH, with opening book enabled
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_TRAD',
#     '-b',  'STOCKFISH',
#     '-m',  'B',
#     '-dw', '4',
#     '-dbs','5', '-sb','3',
#     '-ob',
#     '-sp', $StockfishPath,
#     '-g',  "game_minimax_trad_vs_sf_ob_$RunTag.txt",
#     '-l',  "log_minimax_trad_vs_sf_ob_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 13: MINIMAX_NN vs STOCKFISH, with opening book enabled
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_NN',
#     '-b',  'STOCKFISH',
#     '-m',  'B',
#     '-dw', '3',
#     '-dbs','5', '-sb','3',
#     '-ob',
#     '-sp', $StockfishPath,
#     '-g',  "game_minimax_nn_vs_sf_ob_$RunTag.txt",
#     '-l',  "log_minimax_nn_vs_sf_ob_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 14: MCTS_TRAD (1s) vs MINIMAX_TRAD (depth 4)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MINIMAX_TRAD',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-db', '4',
#     '-g',  "game_mcts_trad_vs_minimax_trad_$RunTag.txt",
#     '-l',  "log_mcts_trad_vs_minimax_trad_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 15: MINIMAX_TRAD (depth 4) vs MCTS_TRAD (1s)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-dw', '4',
#     '-mtw','1', '-mtb','1',
#     '-g',  "game_minimax_trad_vs_mcts_trad_$RunTag.txt",
#     '-l',  "log_minimax_trad_vs_mcts_trad_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 16: MCTS_NN (1s) vs MINIMAX_NN (depth 3)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_NN',
#     '-b',  'MINIMAX_NN',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-db', '3',
#     '-g',  "game_mcts_nn_vs_minimax_nn_$RunTag.txt",
#     '-l',  "log_mcts_nn_vs_minimax_nn_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 17: MINIMAX_NN (depth 3) vs MCTS_NN (1s)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_NN',
#     '-b',  'MCTS_NN',
#     '-m',  'B',
#     '-dw', '3',
#     '-mtw','1', '-mtb','1',
#     '-g',  "game_minimax_nn_vs_mcts_nn_$RunTag.txt",
#     '-l',  "log_minimax_nn_vs_mcts_nn_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 18: MCTS_TRAD (1s) vs MCTS_NN (1s) - evaluator comparison
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_NN',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-g',  "game_mcts_trad_vs_mcts_nn_$RunTag.txt",
#     '-l',  "log_mcts_trad_vs_mcts_nn_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 19: MCTS_NN (1s) vs MCTS_TRAD (1s) - evaluator comparison
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_NN',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-g',  "game_mcts_nn_vs_mcts_trad_$RunTag.txt",
#     '-l',  "log_mcts_nn_vs_mcts_trad_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 20: MINIMAX_TRAD (depth 4) vs MINIMAX_NN (depth 3)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_TRAD',
#     '-b',  'MINIMAX_NN',
#     '-m',  'B',
#     '-dw', '4',
#     '-db', '3',
#     '-g',  "game_minimax_trad_vs_minimax_nn_$RunTag.txt",
#     '-l',  "log_minimax_trad_vs_minimax_nn_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 21: MINIMAX_NN (depth 3) vs MINIMAX_TRAD (depth 4)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_NN',
#     '-b',  'MINIMAX_TRAD',
#     '-m',  'B',
#     '-dw', '3',
#     '-db', '4',
#     '-g',  "game_minimax_nn_vs_minimax_trad_$RunTag.txt",
#     '-l',  "log_minimax_nn_vs_minimax_trad_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 22: MCTS_TRAD self-play sanity (1s vs 1s)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-g',  "game_mcts_trad_selfplay_$RunTag.txt",
#     '-l',  "log_mcts_trad_selfplay_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 23: MCTS_NN self-play sanity (1s vs 1s)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_NN',
#     '-b',  'MCTS_NN',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-g',  "game_mcts_nn_selfplay_$RunTag.txt",
#     '-l',  "log_mcts_nn_selfplay_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 24: MINIMAX_TRAD self-play sanity (depth 4 vs 4)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_TRAD',
#     '-b',  'MINIMAX_TRAD',
#     '-m',  'B',
#     '-dw', '4', '-db','4',
#     '-g',  "game_minimax_trad_selfplay_$RunTag.txt",
#     '-l',  "log_minimax_trad_selfplay_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 25: MINIMAX_NN self-play sanity (depth 3 vs 3)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_NN',
#     '-b',  'MINIMAX_NN',
#     '-m',  'B',
#     '-dw', '3', '-db','3',
#     '-g',  "game_minimax_nn_selfplay_$RunTag.txt",
#     '-l',  "log_minimax_nn_selfplay_$RunTag.txt"
# )

# ============================================================================
# EXECUTION - do not edit below unless you know what you're doing.
# ============================================================================

if (-not $runArgs) {
    Write-Error 'No variant selected - uncomment one of the $runArgs blocks.'
    exit 1
}

# Validate that main.py exists
$mainPath = Join-Path -Path '.' -ChildPath $MainScript
if (-not (Test-Path -LiteralPath $mainPath)) {
    Write-Error "File not found: $mainPath. Are you running this script from the 'engine' folder?"
    exit 1
}

# Ensure out/ exists (main.py writes logs and results there).
$outDir = Join-Path -Path '.' -ChildPath 'out'
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

