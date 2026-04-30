<#
.SYNOPSIS
    Runs the chess engine (main.py) in various predefined MCTS configurations.

.DESCRIPTION
    This script is a simple launcher for testing MCTS variants.
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
# Variant 1: MCTS (trad) vs MCTS (trad), background mode, 1s vs 1s
# Simplest self-play test with the traditional MCTS evaluator.
# ----------------------------------------------------------------------------
#$runArgs = @(
#    '-w',  'MCTS_TRAD',
#    '-b',  'MCTS_TRAD',
#    '-m',  'B',
#    '-mtw', '1', '-mtb', '1',
#    '-g',  "game_mcts_vs_mcts_1s_$RunTag.txt",
#    '-l',  "log_mcts_vs_mcts_1s_$RunTag.txt"
#)

# ----------------------------------------------------------------------------
# Variant 2: MCTS (trad) vs MCTS (trad), GUI, 1s vs 1s, opening book
# For visually observing the game.
# ----------------------------------------------------------------------------
 $runArgs = @(
     '-w',  'MCTS_TRAD',
     '-b',  'MCTS_TRAD',
     '-m',  'G',
     '-mtw', '20', '-mtb', '20',
     '-ob',
     '-g',  "game_mcts_vs_mcts_gui_$RunTag.txt",
     '-l',  "log_mcts_vs_mcts_gui_$RunTag.txt"
 )

# ----------------------------------------------------------------------------
# Variant 3: Asymmetric time — 2s vs 0.5s
# Tests the impact of time budget on playing strength (white is stronger).
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw', '2', '-mtb', '0.5',
#     '-g',  "game_mcts2s_vs_mcts05s_$RunTag.txt",
#     '-l',  "log_mcts2s_vs_mcts05s_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 4: MCTS (trad) vs Stockfish at low skill
# External benchmark. Make sure -sp points to the correct Stockfish binary.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'STOCKFISH',
#     '-m',  'B',
#     '-mtw', '1', '-mtb', '1',
#     '-dbs','5',
#     '-sb', '3',
#     '-sp', 'C:\stockfish\stockfish.exe',
#     '-g',  "game_mcts_vs_sf_$RunTag.txt",
#     '-l',  "log_mcts_vs_sf_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 5: Stockfish vs MCTS (trad) — colors swapped, otherwise same as Variant 4
# Pair with Variant 4 to average out color-based bias.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'STOCKFISH',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-dws','5',
#     '-sw', '3',
#     '-mtw', '1', '-mtb', '1',
#     '-sp', 'C:\stockfish\stockfish.exe',
#     '-g',  "game_sf_vs_mcts_$RunTag.txt",
#     '-l',  "log_sf_vs_mcts_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 6: Start from a custom FEN position
# The FEN file must reside in out/. Useful for tactical/endgame tests.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw', '1', '-mtb', '1',
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
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw', '1', '-mtb', '1',
#     '-d',
#     '-g',  "game_debug_$RunTag.txt",
#     '-l',  "log_debug_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 8: Human vs MCTS (trad) — GUI mode
# For manual sanity-checking of engine play quality.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'H',
#     '-b',  'MCTS_TRAD',
#     '-m',  'G',
#     '-mtw', '1', '-mtb', '1',
#     '-ob',
#     '-g',  "game_human_vs_mcts_$RunTag.txt",
#     '-l',  "log_human_vs_mcts_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 9: MCTS (NN) vs MCTS (NN), background mode, 1s vs 1s
# Neural network evaluator self-play.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_NN',
#     '-b',  'MCTS_NN',
#     '-m',  'B',
#     '-mtw', '1', '-mtb', '1',
#     '-g',  "game_mctsnn_vs_mctsnn_1s_$RunTag.txt",
#     '-l',  "log_mctsnn_vs_mctsnn_1s_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 10: MCTS (NN) vs MCTS (Trad), background mode, 1s vs 1s
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_NN',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw', '1', '-mtb', '1',
#     '-g',  "game_mctsnn_vs_mctstrad_1s_$RunTag.txt",
#     '-l',  "log_mctsnn_vs_mctstrad_1s_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 11: MCTS (Trad) vs MCTS (NN), background mode, 1s vs 1s
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_NN',
#     '-m',  'B',
#     '-mtw', '1', '-mtb', '1',
#     '-g',  "game_mctstrad_vs_mctsnn_1s_$RunTag.txt",
#     '-l',  "log_mctstrad_vs_mctsnn_1s_$RunTag.txt"
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

