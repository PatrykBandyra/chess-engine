<#
.SYNOPSIS
    Runs the chess engine (main.py) in various predefined MCTS configurations.

.DESCRIPTION
    This script is a launcher for testing MCTS variants. To select a variant,
    UNCOMMENT EXACTLY ONE `$runArgs = @(...)` block in the "RUN VARIANTS"
    section, and leave the rest commented out.

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

# Path to the Python interpreter. Defaults to 'python' from PATH.
# For a specific venv/conda env, set the full path, e.g.:
# $Python = 'C:\Users\me\anaconda3\envs\chess-engine\python.exe'
$Python = 'python'

# Entry-point script
$MainScript = 'main.py'

# Common suffix for output filenames - useful for tagging experiment batches
$RunTag = (Get-Date -Format 'yyyyMMdd_HHmmss')

# Stockfish binary path (used by Stockfish-based variants)
$StockfishPath = '..\stockfish_ai\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe'

# ============================================================================
# RUN VARIANTS
# Uncomment EXACTLY ONE of the `$runArgs = @(...)` blocks below.
# ============================================================================

# ----------------------------------------------------------------------------
# Variant 1: MCTS_TRAD vs MCTS_TRAD, background, 1s vs 1s, no opening book
# Simplest self-play test with the traditional MCTS evaluator.
# ----------------------------------------------------------------------------
$runArgs = @(
    '-w',  'MCTS_TRAD',
    '-b',  'MCTS_TRAD',
    '-m',  'B',
    '-mtw','1', '-mtb','1',
    '-g',  "game_mcts_trad_selfplay_1s_$RunTag.txt",
    '-l',  "log_mcts_trad_selfplay_1s_$RunTag.txt"
)

# ----------------------------------------------------------------------------
# Variant 2: MCTS_TRAD vs MCTS_TRAD, GUI, 20s vs 20s, opening book
# For visually observing the game with longer thinking time.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'G',
#     '-mtw','20', '-mtb','20',
#     '-ob',
#     '-g',  "game_mcts_trad_gui_20s_ob_$RunTag.txt",
#     '-l',  "log_mcts_trad_gui_20s_ob_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 3: MCTS_TRAD vs MCTS_TRAD, 3s vs 3s, no opening book
# Longer self-play to surface stability issues with bigger trees.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw','3', '-mtb','3',
#     '-g',  "game_mcts_trad_selfplay_3s_$RunTag.txt",
#     '-l',  "log_mcts_trad_selfplay_3s_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 4: MCTS_TRAD vs MCTS_TRAD, very short 0.2s vs 0.2s
# Stress-test of low time budget (PUCT must cope with few iterations).
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw','0.2', '-mtb','0.2',
#     '-g',  "game_mcts_trad_selfplay_02s_$RunTag.txt",
#     '-l',  "log_mcts_trad_selfplay_02s_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 5: MCTS_TRAD vs MCTS_TRAD, 1s vs 1s, opening book enabled
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-ob',
#     '-g',  "game_mcts_trad_selfplay_1s_ob_$RunTag.txt",
#     '-l',  "log_mcts_trad_selfplay_1s_ob_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 6: MCTS_TRAD vs MCTS_TRAD, 3s vs 3s, opening book enabled
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw','3', '-mtb','3',
#     '-ob',
#     '-g',  "game_mcts_trad_selfplay_3s_ob_$RunTag.txt",
#     '-l',  "log_mcts_trad_selfplay_3s_ob_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 7: Asymmetric time - white 2s, black 0.5s (white advantage)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw','2', '-mtb','0.5',
#     '-g',  "game_mcts_trad_w2s_b05s_$RunTag.txt",
#     '-l',  "log_mcts_trad_w2s_b05s_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 8: Asymmetric time - white 0.5s, black 2s (black advantage)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw','0.5', '-mtb','2',
#     '-g',  "game_mcts_trad_w05s_b2s_$RunTag.txt",
#     '-l',  "log_mcts_trad_w05s_b2s_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 9: Asymmetric time + opening book - white 3s, black 1s, with -ob
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw','3', '-mtb','1',
#     '-ob',
#     '-g',  "game_mcts_trad_w3s_b1s_ob_$RunTag.txt",
#     '-l',  "log_mcts_trad_w3s_b1s_ob_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 10: Debug mode - detailed evaluator timing logs
# Enables timing inside BoardEvaluatorTrad (avg every 200 calls).
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-d',
#     '-g',  "game_mcts_trad_debug_$RunTag.txt",
#     '-l',  "log_mcts_trad_debug_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 11: MCTS_NN vs MCTS_NN, 1s vs 1s
# Neural network evaluator self-play.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_NN',
#     '-b',  'MCTS_NN',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-g',  "game_mcts_nn_selfplay_1s_$RunTag.txt",
#     '-l',  "log_mcts_nn_selfplay_1s_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 12: MCTS_NN vs MCTS_NN, 1s vs 1s, opening book
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_NN',
#     '-b',  'MCTS_NN',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-ob',
#     '-g',  "game_mcts_nn_selfplay_1s_ob_$RunTag.txt",
#     '-l',  "log_mcts_nn_selfplay_1s_ob_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 13: MCTS_NN vs MCTS_NN, asymmetric (white 2s, black 0.5s)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_NN',
#     '-b',  'MCTS_NN',
#     '-m',  'B',
#     '-mtw','2', '-mtb','0.5',
#     '-g',  "game_mcts_nn_w2s_b05s_$RunTag.txt",
#     '-l',  "log_mcts_nn_w2s_b05s_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 14: MCTS_NN (W) vs MCTS_TRAD (B), 1s vs 1s
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
# Variant 15: MCTS_TRAD (W) vs MCTS_NN (B), 1s vs 1s
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
# Variant 16: MCTS_NN (W) vs MCTS_TRAD (B), 1s vs 1s, opening book
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_NN',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-ob',
#     '-g',  "game_mcts_nn_vs_mcts_trad_ob_$RunTag.txt",
#     '-l',  "log_mcts_nn_vs_mcts_trad_ob_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 17: MCTS_TRAD (W, 1s) vs MINIMAX_TRAD (B, depth 4)
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
# Variant 18: MINIMAX_TRAD (W, depth 4) vs MCTS_TRAD (B, 1s)
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
# Variant 19: MCTS_NN (W, 1s) vs MINIMAX_NN (B, depth 3)
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
# Variant 20: MINIMAX_NN (W, depth 3) vs MCTS_NN (B, 1s)
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
# Variant 21: MCTS_TRAD (W, 1s) vs STOCKFISH (B, depth 5, skill 3)
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'STOCKFISH',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-dbs','5', '-sb','3',
#     '-sp', $StockfishPath,
#     '-g',  "game_mcts_trad_vs_sf_$RunTag.txt",
#     '-l',  "log_mcts_trad_vs_sf_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 22: STOCKFISH (W, depth 5, skill 3) vs MCTS_TRAD (B, 1s)
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
# Variant 23: MCTS_NN (W, 1s) vs STOCKFISH (B, depth 5, skill 3)
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
# Variant 24: STOCKFISH (W, depth 5, skill 3) vs MCTS_NN (B, 1s)
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
# Variant 25: MCTS_TRAD vs MCTS_TRAD, 1s vs 1s, start from a custom FEN
# The FEN file must reside in engine/out/. Useful for tactical/endgame tests.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'MCTS_TRAD',
#     '-b',  'MCTS_TRAD',
#     '-m',  'B',
#     '-mtw','1', '-mtb','1',
#     '-i',  'startpos.fen',
#     '-o',  "endpos_$RunTag.fen",
#     '-g',  "game_mcts_trad_fen_$RunTag.txt",
#     '-l',  "log_mcts_trad_fen_$RunTag.txt"
# )

# ----------------------------------------------------------------------------
# Variant 26: Human vs MCTS_TRAD (1s) - GUI mode
# For manual sanity-checking of engine play quality.
# ----------------------------------------------------------------------------
# $runArgs = @(
#     '-w',  'H',
#     '-b',  'MCTS_TRAD',
#     '-m',  'G',
#     '-mtw','1', '-mtb','1',
#     '-ob',
#     '-g',  "game_human_vs_mcts_trad_$RunTag.txt",
#     '-l',  "log_human_vs_mcts_trad_$RunTag.txt"
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

