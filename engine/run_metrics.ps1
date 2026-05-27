<#
.SYNOPSIS
    Runs the chess engine with JSON metrics logging enabled.

.DESCRIPTION
    Launcher that uses the -jl (--json_log) flag to collect per-move metrics
    into a JSONL file. Each line contains move stats (eval, phase, time,
    algorithm-specific counters) and the final line is a game summary.

    Usage:
        .\run_metrics.ps1                     # runs default scenario
        .\run_metrics.ps1 -Scenario mcts_nn   # runs a named scenario
        .\run_metrics.ps1 -List               # lists all available scenarios

    Output goes to engine/out/:
        - game_<scenario>_<timestamp>.txt     : move list
        - log_<scenario>_<timestamp>.txt      : engine log
        - metrics_<scenario>_<timestamp>.jsonl : JSON metrics (per-move + summary)

.NOTES
    Run from the engine/ folder.
#>

param(
    [string]$Scenario = 'mcts_trad_selfplay',
    [switch]$List,
    [switch]$Gui,
    [switch]$OpeningBook,
    [switch]$Debug,
    [switch]$Adjudicate,
    [double]$AdjudicateThreshold = 0.05,
    [int]$AdjudicateMoves = 20
)

# ============================================================================
# COMMON SETTINGS
# ============================================================================

$Python     = 'python'
$MainScript = 'main.py'
$RunTag     = (Get-Date -Format 'yyyyMMdd_HHmmss')

$StockfishPath = '..\stockfish_ai\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe'

$Mode = if ($Gui) { 'G' } else { 'B' }

# ============================================================================
# SCENARIOS
# ============================================================================

$Scenarios = [ordered]@{

    # --- MCTS self-play ---
    'mcts_trad_selfplay' = @{
        Desc = 'MCTS_TRAD vs MCTS_TRAD, 1s/1s'
        Args = @('-w','MCTS_TRAD', '-b','MCTS_TRAD', '-mtw','1', '-mtb','1')
    }
    'mcts_trad_selfplay_3s' = @{
        Desc = 'MCTS_TRAD vs MCTS_TRAD, 3s/3s, opening book'
        Args = @('-w','MCTS_TRAD', '-b','MCTS_TRAD', '-mtw','3', '-mtb','3', '-ob')
    }
    'mcts_nn_selfplay' = @{
        Desc = 'MCTS_NN vs MCTS_NN, 1s/1s'
        Args = @('-w','MCTS_NN', '-b','MCTS_NN', '-mtw','1', '-mtb','1')
    }
    'mcts_nn_selfplay_3s' = @{
        Desc = 'MCTS_NN vs MCTS_NN, 3s/3s, opening book'
        Args = @('-w','MCTS_NN', '-b','MCTS_NN', '-mtw','3', '-mtb','3', '-ob')
    }

    # --- MCTS evaluator comparison ---
    'mcts_trad_vs_nn' = @{
        Desc = 'MCTS_TRAD vs MCTS_NN, 1s/1s'
        Args = @('-w','MCTS_TRAD', '-b','MCTS_NN', '-mtw','1', '-mtb','1')
    }
    'mcts_nn_vs_trad' = @{
        Desc = 'MCTS_NN vs MCTS_TRAD, 1s/1s'
        Args = @('-w','MCTS_NN', '-b','MCTS_TRAD', '-mtw','1', '-mtb','1')
    }

    # --- Minimax self-play ---
    'minimax_trad_selfplay' = @{
        Desc = 'MINIMAX_TRAD vs MINIMAX_TRAD, depth 4'
        Args = @('-w','MINIMAX_TRAD', '-b','MINIMAX_TRAD', '-dw','4', '-db','4')
    }
    'minimax_nn_selfplay' = @{
        Desc = 'MINIMAX_NN vs MINIMAX_NN, depth 3'
        Args = @('-w','MINIMAX_NN', '-b','MINIMAX_NN', '-dw','3', '-db','3')
    }

    # --- Cross-algorithm ---
    'mcts_trad_vs_minimax_trad' = @{
        Desc = 'MCTS_TRAD(1s) vs MINIMAX_TRAD(depth 4)'
        Args = @('-w','MCTS_TRAD', '-b','MINIMAX_TRAD', '-mtw','1', '-mtb','1', '-db','4')
    }
    'minimax_trad_vs_mcts_trad' = @{
        Desc = 'MINIMAX_TRAD(depth 4) vs MCTS_TRAD(1s)'
        Args = @('-w','MINIMAX_TRAD', '-b','MCTS_TRAD', '-dw','4', '-mtw','1', '-mtb','1')
    }
    'mcts_nn_vs_minimax_nn' = @{
        Desc = 'MCTS_NN(1s) vs MINIMAX_NN(depth 3)'
        Args = @('-w','MCTS_NN', '-b','MINIMAX_NN', '-mtw','1', '-mtb','1', '-db','3')
    }
    'minimax_nn_vs_mcts_nn' = @{
        Desc = 'MINIMAX_NN(depth 3) vs MCTS_NN(1s)'
        Args = @('-w','MINIMAX_NN', '-b','MCTS_NN', '-dw','3', '-mtw','1', '-mtb','1')
    }

    # --- vs Stockfish ---
    'mcts_trad_vs_sf' = @{
        Desc = 'MCTS_TRAD(1s) vs STOCKFISH(depth 5, skill 3)'
        Args = @('-w','MCTS_TRAD', '-b','STOCKFISH', '-mtw','1', '-mtb','1',
                 '-dbs','5', '-sb','3', '-sp', $StockfishPath)
    }
    'mcts_nn_vs_sf' = @{
        Desc = 'MCTS_NN(1s) vs STOCKFISH(depth 5, skill 3)'
        Args = @('-w','MCTS_NN', '-b','STOCKFISH', '-mtw','1', '-mtb','1',
                 '-dbs','5', '-sb','3', '-sp', $StockfishPath)
    }
    'minimax_trad_vs_sf' = @{
        Desc = 'MINIMAX_TRAD(depth 4) vs STOCKFISH(depth 5, skill 3)'
        Args = @('-w','MINIMAX_TRAD', '-b','STOCKFISH', '-dw','4',
                 '-dbs','5', '-sb','3', '-sp', $StockfishPath)
    }
    'minimax_nn_vs_sf' = @{
        Desc = 'MINIMAX_NN(depth 3) vs STOCKFISH(depth 5, skill 3)'
        Args = @('-w','MINIMAX_NN', '-b','STOCKFISH', '-dw','3',
                 '-dbs','5', '-sb','3', '-sp', $StockfishPath)
    }
}

# ============================================================================
# LIST MODE
# ============================================================================

if ($List) {
    Write-Host "`nAvailable scenarios:" -ForegroundColor Cyan
    Write-Host ('-' * 60)
    foreach ($key in $Scenarios.Keys) {
        Write-Host ("  {0,-35} {1}" -f $key, $Scenarios[$key].Desc)
    }
    Write-Host ('-' * 60)
    Write-Host "Usage: .\run_metrics.ps1 -Scenario <name> [-Gui] [-OpeningBook] [-Debug] [-Adjudicate]`n"
    exit 0
}

# ============================================================================
# VALIDATE
# ============================================================================

if (-not $Scenarios.Contains($Scenario)) {
    Write-Error "Unknown scenario: '$Scenario'. Use -List to see available options."
    exit 1
}

$mainPath = Join-Path -Path '.' -ChildPath $MainScript
if (-not (Test-Path -LiteralPath $mainPath)) {
    Write-Error "File not found: $mainPath. Run this script from the 'engine' folder."
    exit 1
}

$outDir = Join-Path -Path '.' -ChildPath 'out'
if (-not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
    Write-Host "Created directory: $outDir"
}

# ============================================================================
# BUILD ARGUMENTS
# ============================================================================

$cfg = $Scenarios[$Scenario]

$gameFile    = "game_${Scenario}_${RunTag}.txt"
$logFile     = "log_${Scenario}_${RunTag}.txt"
$metricsFile = "metrics_${Scenario}_${RunTag}.jsonl"

$runArgs = @()
$runArgs += $cfg.Args
$runArgs += @('-m', $Mode)
$runArgs += @('-g', $gameFile)
$runArgs += @('-l', $logFile)
$runArgs += @('-jl', $metricsFile)

if ($OpeningBook) {
    $runArgs += '-ob'
}
if ($Debug) {
    $runArgs += '-d'
}
if ($Adjudicate) {
    $runArgs += @('-adj', '-adjt', "$AdjudicateThreshold", '-adjm', "$AdjudicateMoves")
}

# ============================================================================
# RUN
# ============================================================================

Write-Host ''
Write-Host '============================================================' -ForegroundColor Green
Write-Host "  Scenario:  $Scenario" -ForegroundColor Green
Write-Host "  Desc:      $($cfg.Desc)" -ForegroundColor Green
Write-Host "  Mode:      $Mode" -ForegroundColor Green
Write-Host "  Metrics:   out/$metricsFile" -ForegroundColor Yellow
Write-Host "  Game log:  out/$gameFile"
Write-Host "  Engine log: out/$logFile"
Write-Host '============================================================' -ForegroundColor Green
Write-Host "Command: $Python $MainScript $($runArgs -join ' ')"
Write-Host ''

& $Python $MainScript @runArgs
$exitCode = $LASTEXITCODE

Write-Host ''
Write-Host '============================================================'
Write-Host "Finished with exit code: $exitCode"

if ($exitCode -eq 0 -and (Test-Path -LiteralPath "out/$metricsFile")) {
    $lineCount = (Get-Content "out/$metricsFile" | Measure-Object -Line).Lines
    $fileSize  = (Get-Item "out/$metricsFile").Length
    Write-Host "Metrics: $lineCount records, $([math]::Round($fileSize / 1KB, 1)) KB" -ForegroundColor Cyan

    Write-Host ''
    Write-Host 'Game summary:' -ForegroundColor Cyan
    Get-Content "out/$metricsFile" -Tail 1 | ConvertFrom-Json | Format-List
}

exit $exitCode
