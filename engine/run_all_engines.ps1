<#
.SYNOPSIS
    Runs the chess engine (main.py) through a battery of cross-engine
    matchups: MCTS_TRAD / MCTS_NN / MINIMAX_TRAD / MINIMAX_NN vs STOCKFISH,
    plus engine-vs-engine comparisons.

.DESCRIPTION
    Batch runner mirroring run_mcts.ps1: defines a table of scenarios and
    runs them all back-to-back, with per-scenario game and log files in
    engine/out/ tagged by the batch RunTag.

    Coverage groups:
        1) <Engine> vs STOCKFISH and STOCKFISH vs <Engine> for every engine
           (MCTS_TRAD, MCTS_NN, MINIMAX_TRAD, MINIMAX_NN), with low Stockfish
           skill so the pure-engine side has a realistic chance.
        2) Same matchups but with the opening book enabled (-ob) to test
           the book integration paired with each engine.
        3) Engine vs engine cross-matches (no Stockfish): MCTS vs Minimax,
           Trad vs NN, both color assignments.
        4) Sanity self-play games (one per engine) to detect regressions
           that would only show up against itself.

    Each scenario carries explicit time / depth budgets so the comparison
    against Stockfish is reproducible after the recent MCTS fixes.

.PARAMETER Filter
    Wildcard filter on scenario Name (e.g. '*_vs_sf*', 'mcts_*_ob*').
    Default: '*' (all scenarios).

.PARAMETER ListOnly
    Print the scenario table and exit without running anything.

.PARAMETER DryRun
    Print the resolved command line for each matching scenario without launching it.

.PARAMETER StopOnError
    Abort the batch as soon as a scenario exits with a non-zero code.

.PARAMETER StockfishPath
    Override the default path to the Stockfish binary. Stockfish-dependent
    scenarios are skipped automatically if the binary is not found.

.PARAMETER Python
    Python interpreter to use. Default: 'python' from PATH.

.PARAMETER MctsTime
    MCTS time budget in seconds (per side) used by every MCTS scenario.
    Default: 1.

.PARAMETER MinimaxDepth
    Minimax search depth used by every MINIMAX_TRAD scenario. Default: 4.

.PARAMETER MinimaxNNDepth
    Minimax search depth used by every MINIMAX_NN scenario (typically lower
    than MinimaxDepth because NN evaluation is more expensive). Default: 3.

.PARAMETER StockfishDepth
    Stockfish search depth used by every Stockfish-side scenario. Default: 5.

.PARAMETER StockfishSkill
    Stockfish skill level (0..20) used by every Stockfish-side scenario.
    Lower = weaker. Default: 3.

.EXAMPLE
    .\run_all_engines.ps1 -ListOnly

.EXAMPLE
    # Only matchups against Stockfish
    .\run_all_engines.ps1 -Filter '*_vs_sf*','sf_vs_*'

.EXAMPLE
    # Compare engines after a tuning change, abort on first failure
    .\run_all_engines.ps1 -Filter 'mcts_trad_vs_*' -StopOnError

.NOTES
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

    Run from the engine/ folder.
#>

[CmdletBinding()]
param(
    [string[]] $Filter         = @('*'),
    [switch]   $ListOnly,
    [switch]   $DryRun,
    [switch]   $StopOnError,
    [string]   $StockfishPath  = '..\stockfish_ai\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe',
    [string]   $Python         = 'python',
    [double]   $MctsTime       = 1.0,
    [int]      $MinimaxDepth   = 4,
    [int]      $MinimaxNNDepth = 3,
    [int]      $StockfishDepth = 5,
    [int]      $StockfishSkill = 3
)

# ============================================================================
# COMMON SETTINGS
# ============================================================================
$MainScript = 'main.py'
$RunTag     = (Get-Date -Format 'yyyyMMdd_HHmmss')

# Convenience strings reused below.
$mt   = "$MctsTime"
$dT   = "$MinimaxDepth"
$dNN  = "$MinimaxNNDepth"
$sfD  = "$StockfishDepth"
$sfS  = "$StockfishSkill"

# ============================================================================
# SCENARIO TABLE
# Fields: Name, Desc, NeedsSF (bool), Args (string[])
# Game/log filenames are appended automatically using $RunTag.
# ============================================================================
$Scenarios = @(

    # ------------------------------------------------------------------------
    # 1) Engines vs Stockfish — both color assignments
    # ------------------------------------------------------------------------
    @{
        Name    = 'mcts_trad_vs_sf'
        Desc    = "MCTS_TRAD (W, ${mt}s) vs STOCKFISH (B, depth $sfD, skill $sfS)"
        NeedsSF = $true
        Args    = @('-w','MCTS_TRAD','-b','STOCKFISH','-m','B','-mtw',$mt,'-mtb',$mt,'-dbs',$sfD,'-sb',$sfS)
    },
    @{
        Name    = 'sf_vs_mcts_trad'
        Desc    = "STOCKFISH (W, depth $sfD, skill $sfS) vs MCTS_TRAD (B, ${mt}s)"
        NeedsSF = $true
        Args    = @('-w','STOCKFISH','-b','MCTS_TRAD','-m','B','-dws',$sfD,'-sw',$sfS,'-mtw',$mt,'-mtb',$mt)
    },
    @{
        Name    = 'mcts_nn_vs_sf'
        Desc    = "MCTS_NN (W, ${mt}s) vs STOCKFISH (B, depth $sfD, skill $sfS)"
        NeedsSF = $true
        Args    = @('-w','MCTS_NN','-b','STOCKFISH','-m','B','-mtw',$mt,'-mtb',$mt,'-dbs',$sfD,'-sb',$sfS)
    },
    @{
        Name    = 'sf_vs_mcts_nn'
        Desc    = "STOCKFISH (W, depth $sfD, skill $sfS) vs MCTS_NN (B, ${mt}s)"
        NeedsSF = $true
        Args    = @('-w','STOCKFISH','-b','MCTS_NN','-m','B','-dws',$sfD,'-sw',$sfS,'-mtw',$mt,'-mtb',$mt)
    },
    @{
        Name    = 'minimax_trad_vs_sf'
        Desc    = "MINIMAX_TRAD (W, depth $dT) vs STOCKFISH (B, depth $sfD, skill $sfS)"
        NeedsSF = $true
        Args    = @('-w','MINIMAX_TRAD','-b','STOCKFISH','-m','B','-dw',$dT,'-dbs',$sfD,'-sb',$sfS)
    },
    @{
        Name    = 'sf_vs_minimax_trad'
        Desc    = "STOCKFISH (W, depth $sfD, skill $sfS) vs MINIMAX_TRAD (B, depth $dT)"
        NeedsSF = $true
        Args    = @('-w','STOCKFISH','-b','MINIMAX_TRAD','-m','B','-dws',$sfD,'-sw',$sfS,'-db',$dT)
    },
    @{
        Name    = 'minimax_nn_vs_sf'
        Desc    = "MINIMAX_NN (W, depth $dNN) vs STOCKFISH (B, depth $sfD, skill $sfS)"
        NeedsSF = $true
        Args    = @('-w','MINIMAX_NN','-b','STOCKFISH','-m','B','-dw',$dNN,'-dbs',$sfD,'-sb',$sfS)
    },
    @{
        Name    = 'sf_vs_minimax_nn'
        Desc    = "STOCKFISH (W, depth $sfD, skill $sfS) vs MINIMAX_NN (B, depth $dNN)"
        NeedsSF = $true
        Args    = @('-w','STOCKFISH','-b','MINIMAX_NN','-m','B','-dws',$sfD,'-sw',$sfS,'-db',$dNN)
    },

    # ------------------------------------------------------------------------
    # 2) Same matchups with opening book enabled (book integration check)
    # ------------------------------------------------------------------------
    @{
        Name    = 'mcts_trad_vs_sf_ob'
        Desc    = "MCTS_TRAD (W, ${mt}s) vs STOCKFISH (B, depth $sfD, skill $sfS) + opening book"
        NeedsSF = $true
        Args    = @('-w','MCTS_TRAD','-b','STOCKFISH','-m','B','-mtw',$mt,'-mtb',$mt,'-dbs',$sfD,'-sb',$sfS,'-ob')
    },
    @{
        Name    = 'sf_vs_mcts_trad_ob'
        Desc    = "STOCKFISH (W, depth $sfD, skill $sfS) vs MCTS_TRAD (B, ${mt}s) + opening book"
        NeedsSF = $true
        Args    = @('-w','STOCKFISH','-b','MCTS_TRAD','-m','B','-dws',$sfD,'-sw',$sfS,'-mtw',$mt,'-mtb',$mt,'-ob')
    },
    @{
        Name    = 'mcts_nn_vs_sf_ob'
        Desc    = "MCTS_NN (W, ${mt}s) vs STOCKFISH (B, depth $sfD, skill $sfS) + opening book"
        NeedsSF = $true
        Args    = @('-w','MCTS_NN','-b','STOCKFISH','-m','B','-mtw',$mt,'-mtb',$mt,'-dbs',$sfD,'-sb',$sfS,'-ob')
    },
    @{
        Name    = 'minimax_trad_vs_sf_ob'
        Desc    = "MINIMAX_TRAD (W, depth $dT) vs STOCKFISH (B, depth $sfD, skill $sfS) + opening book"
        NeedsSF = $true
        Args    = @('-w','MINIMAX_TRAD','-b','STOCKFISH','-m','B','-dw',$dT,'-dbs',$sfD,'-sb',$sfS,'-ob')
    },
    @{
        Name    = 'minimax_nn_vs_sf_ob'
        Desc    = "MINIMAX_NN (W, depth $dNN) vs STOCKFISH (B, depth $sfD, skill $sfS) + opening book"
        NeedsSF = $true
        Args    = @('-w','MINIMAX_NN','-b','STOCKFISH','-m','B','-dw',$dNN,'-dbs',$sfD,'-sb',$sfS,'-ob')
    },

    # ------------------------------------------------------------------------
    # 3) Engine vs engine cross-matches (no Stockfish required)
    # ------------------------------------------------------------------------
    @{
        Name = 'mcts_trad_vs_minimax_trad'
        Desc = "MCTS_TRAD (W, ${mt}s) vs MINIMAX_TRAD (B, depth $dT)"
        Args = @('-w','MCTS_TRAD','-b','MINIMAX_TRAD','-m','B','-mtw',$mt,'-mtb',$mt,'-db',$dT)
    },
    @{
        Name = 'minimax_trad_vs_mcts_trad'
        Desc = "MINIMAX_TRAD (W, depth $dT) vs MCTS_TRAD (B, ${mt}s)"
        Args = @('-w','MINIMAX_TRAD','-b','MCTS_TRAD','-m','B','-dw',$dT,'-mtw',$mt,'-mtb',$mt)
    },
    @{
        Name = 'mcts_nn_vs_minimax_nn'
        Desc = "MCTS_NN (W, ${mt}s) vs MINIMAX_NN (B, depth $dNN)"
        Args = @('-w','MCTS_NN','-b','MINIMAX_NN','-m','B','-mtw',$mt,'-mtb',$mt,'-db',$dNN)
    },
    @{
        Name = 'minimax_nn_vs_mcts_nn'
        Desc = "MINIMAX_NN (W, depth $dNN) vs MCTS_NN (B, ${mt}s)"
        Args = @('-w','MINIMAX_NN','-b','MCTS_NN','-m','B','-dw',$dNN,'-mtw',$mt,'-mtb',$mt)
    },
    @{
        Name = 'mcts_trad_vs_mcts_nn'
        Desc = "MCTS_TRAD (W, ${mt}s) vs MCTS_NN (B, ${mt}s) — eval comparison"
        Args = @('-w','MCTS_TRAD','-b','MCTS_NN','-m','B','-mtw',$mt,'-mtb',$mt)
    },
    @{
        Name = 'mcts_nn_vs_mcts_trad'
        Desc = "MCTS_NN (W, ${mt}s) vs MCTS_TRAD (B, ${mt}s) — eval comparison"
        Args = @('-w','MCTS_NN','-b','MCTS_TRAD','-m','B','-mtw',$mt,'-mtb',$mt)
    },
    @{
        Name = 'minimax_trad_vs_minimax_nn'
        Desc = "MINIMAX_TRAD (W, depth $dT) vs MINIMAX_NN (B, depth $dNN)"
        Args = @('-w','MINIMAX_TRAD','-b','MINIMAX_NN','-m','B','-dw',$dT,'-db',$dNN)
    },
    @{
        Name = 'minimax_nn_vs_minimax_trad'
        Desc = "MINIMAX_NN (W, depth $dNN) vs MINIMAX_TRAD (B, depth $dT)"
        Args = @('-w','MINIMAX_NN','-b','MINIMAX_TRAD','-m','B','-dw',$dNN,'-db',$dT)
    },

    # ------------------------------------------------------------------------
    # 4) Self-play sanity games (one per engine)
    # ------------------------------------------------------------------------
    @{
        Name = 'mcts_trad_selfplay'
        Desc = "MCTS_TRAD self-play, ${mt}s vs ${mt}s"
        Args = @('-w','MCTS_TRAD','-b','MCTS_TRAD','-m','B','-mtw',$mt,'-mtb',$mt)
    },
    @{
        Name = 'mcts_nn_selfplay'
        Desc = "MCTS_NN self-play, ${mt}s vs ${mt}s"
        Args = @('-w','MCTS_NN','-b','MCTS_NN','-m','B','-mtw',$mt,'-mtb',$mt)
    },
    @{
        Name = 'minimax_trad_selfplay'
        Desc = "MINIMAX_TRAD self-play, depth $dT vs $dT"
        Args = @('-w','MINIMAX_TRAD','-b','MINIMAX_TRAD','-m','B','-dw',$dT,'-db',$dT)
    },
    @{
        Name = 'minimax_nn_selfplay'
        Desc = "MINIMAX_NN self-play, depth $dNN vs $dNN"
        Args = @('-w','MINIMAX_NN','-b','MINIMAX_NN','-m','B','-dw',$dNN,'-db',$dNN)
    }
)

# ============================================================================
# EXECUTION
# ============================================================================

# Sanity: must be run from engine/ (where main.py lives).
if (-not (Test-Path -LiteralPath $MainScript)) {
    Write-Error "File not found: $MainScript. Run this script from the 'engine' folder."
    exit 1
}

# Ensure out/ exists (main.py writes logs and results there).
$outDir = Join-Path -Path '.' -ChildPath 'out'
if (-not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
    Write-Host "Created directory: $outDir"
}

$stockfishAvailable = Test-Path -LiteralPath $StockfishPath

# Filter scenarios: a scenario is selected if its Name matches ANY of the
# patterns supplied via -Filter. Default array is @('*') -> everything.
$selected = @($Scenarios | Where-Object {
    $name = $_.Name
    foreach ($pat in $Filter) {
        if ($name -like $pat) { return $true }
    }
    return $false
})

if (-not $selected -or $selected.Count -eq 0) {
    Write-Error "No scenarios matched filter(s): $($Filter -join ', ')."
    exit 1
}

if ($ListOnly) {
    Write-Host '============================================================'
    Write-Host "Scenarios matching filter(s) [$($Filter -join ', ')]:"
    Write-Host '============================================================'
    foreach ($s in $selected) {
        if ($s.NeedsSF) { $tag = '  [needs-stockfish]' } else { $tag = '' }
        Write-Host ('  - {0,-32} {1}{2}' -f $s.Name, $s.Desc, $tag)
    }
    Write-Host ''
    Write-Host "Stockfish binary: $StockfishPath  (available: $stockfishAvailable)"
    Write-Host "Defaults: MctsTime=${MctsTime}s  MinimaxDepth=$MinimaxDepth  MinimaxNNDepth=$MinimaxNNDepth  StockfishDepth=$StockfishDepth  StockfishSkill=$StockfishSkill"
    exit 0
}

Write-Host '============================================================'
Write-Host "All-engines scenario batch — RunTag: $RunTag"
Write-Host "Filter(s): $($Filter -join ', ')   Scenarios selected: $($selected.Count)"
Write-Host "Stockfish available: $stockfishAvailable  ($StockfishPath)"
Write-Host "Defaults: MctsTime=${MctsTime}s  MinimaxDepth=$MinimaxDepth  MinimaxNNDepth=$MinimaxNNDepth  StockfishDepth=$StockfishDepth  StockfishSkill=$StockfishSkill"
Write-Host '============================================================'

$results = New-Object System.Collections.Generic.List[object]
$index   = 0

foreach ($s in $selected) {
    $index++

    if ($s.NeedsSF -and -not $stockfishAvailable) {
        Write-Host ''
        Write-Host "[$index/$($selected.Count)] SKIP $($s.Name) — Stockfish binary not found."
        $results.Add([pscustomobject]@{ Name = $s.Name; Status = 'SKIPPED'; ExitCode = $null; Seconds = 0.0 })
        continue
    }

    # Build final argv: copy scenario args, add stockfish path if needed,
    # and append per-scenario game/log filenames tagged with the batch RunTag.
    $argv = @($s.Args)
    if ($s.NeedsSF) {
        $argv += @('-sp', $StockfishPath)
    }
    $argv += @(
        '-g', "game_$($s.Name)_$RunTag.txt",
        '-l', "log_$($s.Name)_$RunTag.txt"
    )

    Write-Host ''
    Write-Host '------------------------------------------------------------'
    Write-Host "[$index/$($selected.Count)] $($s.Name)"
    Write-Host "    $($s.Desc)"
    Write-Host "    Cmd: $Python $MainScript $($argv -join ' ')"
    Write-Host '------------------------------------------------------------'

    if ($DryRun) {
        $results.Add([pscustomobject]@{ Name = $s.Name; Status = 'DRYRUN'; ExitCode = $null; Seconds = 0.0 })
        continue
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Python $MainScript @argv
    $exitCode = $LASTEXITCODE
    $sw.Stop()
    $elapsed = [Math]::Round($sw.Elapsed.TotalSeconds, 2)

    if ($exitCode -eq 0) { $status = 'OK' } else { $status = 'FAIL' }
    Write-Host "    -> $status (exit=$exitCode, elapsed=${elapsed}s)"
    $results.Add([pscustomobject]@{
        Name     = $s.Name
        Status   = $status
        ExitCode = $exitCode
        Seconds  = $elapsed
    })

    if ($StopOnError -and $exitCode -ne 0) {
        Write-Host ''
        Write-Warning "StopOnError set — aborting batch after failing scenario '$($s.Name)'."
        break
    }
}

# ============================================================================
# SUMMARY
# ============================================================================
Write-Host ''
Write-Host '============================================================'
Write-Host 'Batch summary'
Write-Host '============================================================'
$results | Format-Table -AutoSize

$failed = ($results | Where-Object { $_.Status -eq 'FAIL' }).Count
if ($failed -gt 0) {
    Write-Host ''
    Write-Warning "$failed scenario(s) failed."
    exit 1
}

Write-Host ''
Write-Host 'All selected scenarios completed successfully.'
exit 0

