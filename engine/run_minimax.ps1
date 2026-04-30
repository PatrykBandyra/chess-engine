<#
.SYNOPSIS
    Runs the chess engine (engine/main.py) in various predefined Minimax configurations.

.DESCRIPTION
    This script is a simple launcher for testing Minimax variants.
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
        -mtw / --mcts_time_white       : MCTS time budget in seconds for white
        -mtb / --mcts_time_black       : MCTS time budget in seconds for black
        -d  / --debug                  : debug mode (flag)

.NOTES
    The script assumes the repository layout:
        engine/main.py
        engine/out/         <- output directory (auto-created if missing)
    Python is launched with engine/ as the working directory because the
    in-engine imports are relative to that folder.
#>

# ============================================================================
# COMMON SETTINGS (edit here if your environment differs)
# ============================================================================

# Path to the Python interpreter. Defaults to 'python' from PATH.
# For a specific venv, set the full path, e.g.:
# $Python = 'C:\Users\me\.venvs\chess\Scripts\python.exe'
$Python = 'python'

# Entry-point script
$MainScript = 'main.py'

# Common suffix for output filenames — useful for tagging experiment batches
$RunTag = (Get-Date -Format 'yyyyMMdd_HHmmss')

# ============================================================================
# RUN VARIANTS
# Uncomment EXACTLY ONE of the `$runArgs = @(...)` blocks below.
# ============================================================================

# ----------------------------------------------------------------------------
# Variant 1: Minimax (trad) vs Minimax (trad), background mode, depth 4 vs 4
# Simplest self-play test with the traditional evaluator.
# ----------------------------------------------------------------------------
#$runArgs = @(
#    '-w',  'MINIMAX_TRAD',
#    '-b',  'MINIMAX_TRAD',
#    '-m',  'B',
#    '-dw', '4',
#    '-db', '4',
#    '-g',  "game_trad_vs_trad_d4_$RunTag.txt",
#    '-l',  "log_trad_vs_trad_d4_$RunTag.txt"
#)

# ----------------------------------------------------------------------------
# Variant 2: Minimax (trad) vs Minimax (trad), GUI, depth 4 vs 4, opening book
# For visually observing the game.
# ----------------------------------------------------------------------------
 $runArgs = @(
     '-w',  'MINIMAX_TRAD',
     '-b',  'MINIMAX_TRAD',
     '-m',  'G',
     '-dw', '4',
     '-db', '4',
     '-ob',
     '-g',  "game_trad_vs_trad_gui_$RunTag.txt",
     '-l',  "log_trad_vs_trad_gui_$RunTag.txt"
 )

# ----------------------------------------------------------------------------
# Variant 3: Asymmetric depth — d=5 vs d=3
# Tests the impact of depth on playing strength (white is stronger).
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_TRAD',
#     '-b',  'MINIMAX_TRAD',
#     '-m',  'G',
#     '-dw', '5',
#     '-db', '3',
#     '-ob',
#     '-g',  "game_d5_vs_d3_$RunTag.txt",
#     '-l',  "log_d5_vs_d3_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 4: Minimax (trad) vs Stockfish at low skill
# External benchmark. Make sure -sp points to the correct Stockfish binary.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_TRAD',
#     '-b',  'STOCKFISH',
#     '-m',  'G',
#     '-dw', '4',
#     '-dbs','5',
#     '-sb', '3',
#     '-sp', '../stockfish_ai/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe',
#     '-g',  "game_trad_vs_sf_$RunTag.txt",
#     '-l',  "log_trad_vs_sf_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 5: Stockfish vs Minimax (trad) — colors swapped, otherwise same as Variant 4
# Pair with Variant 4 to average out color-based bias.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'STOCKFISH',
#     '-b',  'MINIMAX_TRAD',
#     '-m',  'G',
#     '-ob',
#     '-dws','5',
#     '-sw', '3',
#     '-db', '4',
#     '-sp', '../stockfish_ai/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe',
#     '-g',  "game_sf_vs_trad_$RunTag.txt",
#     '-l',  "log_sf_vs_trad_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 6: Start from a custom FEN position
# The FEN file must reside in engine/out/. Useful for tactical/endgame tests.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_TRAD',
#     '-b',  'MINIMAX_TRAD',
#     '-m',  'B',
#     '-dw', '5',
#     '-db', '5',
#     '-i',  'startpos.fen',
#     '-o',  "endpos_$RunTag.fen",
#     '-g',  "game_fen_$RunTag.txt",
#     '-l',  "log_fen_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 7: Debug mode — detailed evaluator timing logs
# Enables timing inside BoardEvaluatorTrad (avg every 200 calls).
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_TRAD',
#     '-b',  'MINIMAX_TRAD',
#     '-m',  'B',
#     '-dw', '4',
#     '-db', '4',
#     '-d',
#     '-g',  "game_debug_$RunTag.txt",
#     '-l',  "log_debug_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 8: Human vs Minimax (trad) — GUI mode
# For manual sanity-checking of engine play quality.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'H',
#     '-b',  'MINIMAX_TRAD',
#     '-m',  'G',
#     '-db', '4',
#     '-ob',
#     '-g',  "game_human_vs_trad_$RunTag.txt",
#     '-l',  "log_human_vs_trad_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 9: Minimax (NN) vs Minimax (NN), background mode, depth 3 vs 3
# Neural network evaluator self-play.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_NN',
#     '-b',  'MINIMAX_NN',
#     '-m',  'B',
#     '-dw', '3',
#     '-db', '3',
#     '-g',  "game_nn_vs_nn_d3_$RunTag.txt",
#     '-l',  "log_nn_vs_nn_d3_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 10: Minimax (NN) vs Minimax (Trad), background mode, depth 3 vs 4
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
# Variant 11: Minimax (Trad) vs Minimax (NN), background mode, depth 4 vs 3
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MINIMAX_TRAD',
#     '-b',  'MINIMAX_NN',
#     '-m',  'G',
#     '-ob',
#     '-dw', '4',
#     '-db', '3',
#     '-g',  "game_trad_vs_nn_$RunTag.txt",
#     '-l',  "log_trad_vs_nn_$RunTag.txt"
# )

# ============================================================================
# EXECUTION — do not edit below unless you know what you're doing.
# ============================================================================

if (-not $runArgs) {
    Write-Error 'No variant selected — uncomment one of the $runArgs blocks.'
    exit 1
}

# Validate that main.py exists
$mainPath = Join-Path -Path "." -ChildPath $MainScript
if (-not (Test-Path -LiteralPath $mainPath)) {
    Write-Error "File not found: $mainPath. Are you running this script from the 'engine' folder?"
    exit 1
}

# Ensure out/ exists (main.py writes logs and results there).
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
