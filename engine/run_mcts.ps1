<#
.SYNOPSIS
    Runs the chess engine (main.py) through a battery of predefined MCTS
    test scenarios in sequence.

.DESCRIPTION
    Unlike the older "uncomment one block" launcher, this script defines a
    table of MCTS scenarios and runs them all back-to-back. Every scenario
    produces its own game / log file in engine/out/ tagged with the batch
    timestamp and the scenario name, so results can be compared after the
    batch finishes.

    Coverage includes:
        * MCTS_TRAD vs MCTS_TRAD (with/without opening book, equal times)
        * Asymmetric MCTS time budgets per color (white > black, black > white)
        * Very short and longer MCTS budgets
        * Debug mode (evaluator timing)
        * MCTS_NN vs MCTS_NN, MCTS_NN vs MCTS_TRAD (both sides)
        * MCTS_TRAD/MCTS_NN vs MINIMAX_TRAD/MINIMAX_NN (both sides)
        * MCTS_TRAD/MCTS_NN vs STOCKFISH (both sides)
        * Custom FEN start position (only if the FEN file exists in out/)

.PARAMETER Filter
    Only run scenarios whose Name matches this wildcard pattern (e.g. 'mcts_*_ob').
    Default: '*' (all scenarios).

.PARAMETER ListOnly
    Print the scenario table and exit without running anything.

.PARAMETER DryRun
    Print the resolved command line for each matching scenario without launching it.

.PARAMETER StopOnError
    Abort the batch as soon as a scenario exits with a non-zero code.
    Default: continue with the remaining scenarios.

.PARAMETER StockfishPath
    Override the default path to the Stockfish binary used by Stockfish-based
    scenarios. If the binary is missing, those scenarios are skipped.

.PARAMETER Python
    Python interpreter to use. Default: 'python' from PATH.

.EXAMPLE
    .\run_mcts.ps1 -ListOnly

.EXAMPLE
    # Run only opening-book scenarios
    .\run_mcts.ps1 -Filter '*_ob*'

.EXAMPLE
    # Quick smoke run, abort on first failure
    .\run_mcts.ps1 -Filter 'mcts_trad_*' -StopOnError

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

    The script must be run from the engine/ folder (where main.py lives).
#>

[CmdletBinding()]
param(
    [string]   $Filter        = '*',
    [switch]   $ListOnly,
    [switch]   $DryRun,
    [switch]   $StopOnError,
    [string]   $StockfishPath = '..\stockfish_ai\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe',
    [string]   $Python        = 'python'
)

# ============================================================================
# COMMON SETTINGS
# ============================================================================
$MainScript = 'main.py'
$RunTag     = (Get-Date -Format 'yyyyMMdd_HHmmss')

# Optional input FEN scenario — only included when this file exists in out/.
$FenInput = 'startpos.fen'

# ============================================================================
# SCENARIO TABLE
# Each scenario is a hashtable:
#   Name      : short identifier used in output filenames and -Filter matching
#   Desc      : free-text description (printed before running)
#   NeedsSF   : $true if this scenario depends on Stockfish being available
#   NeedsFen  : $true if this scenario depends on $FenInput existing in out/
#   Args      : raw argv passed to main.py (game/log filenames are appended below)
# ============================================================================
$Scenarios = @(
    # ------------------------------------------------------------------------
    # 1) Baselines: MCTS_TRAD self-play, equal times, no opening book
    # ------------------------------------------------------------------------
    @{
        Name = 'mcts_trad_selfplay_1s'
        Desc = 'MCTS_TRAD vs MCTS_TRAD, 1s vs 1s, no opening book'
        Args = @('-w','MCTS_TRAD','-b','MCTS_TRAD','-m','B','-mtw','1','-mtb','1')
    },
    @{
        Name = 'mcts_trad_selfplay_3s'
        Desc = 'MCTS_TRAD vs MCTS_TRAD, 3s vs 3s, no opening book'
        Args = @('-w','MCTS_TRAD','-b','MCTS_TRAD','-m','B','-mtw','3','-mtb','3')
    },
    @{
        Name = 'mcts_trad_selfplay_short_02s'
        Desc = 'MCTS_TRAD vs MCTS_TRAD, 0.2s vs 0.2s — stress-test of low budget'
        Args = @('-w','MCTS_TRAD','-b','MCTS_TRAD','-m','B','-mtw','0.2','-mtb','0.2')
    },

    # ------------------------------------------------------------------------
    # 2) Opening-book toggle
    # ------------------------------------------------------------------------
    @{
        Name = 'mcts_trad_selfplay_1s_ob'
        Desc = 'MCTS_TRAD vs MCTS_TRAD, 1s vs 1s, opening book enabled'
        Args = @('-w','MCTS_TRAD','-b','MCTS_TRAD','-m','B','-mtw','1','-mtb','1','-ob')
    },
    @{
        Name = 'mcts_trad_selfplay_3s_ob'
        Desc = 'MCTS_TRAD vs MCTS_TRAD, 3s vs 3s, opening book enabled'
        Args = @('-w','MCTS_TRAD','-b','MCTS_TRAD','-m','B','-mtw','3','-mtb','3','-ob')
    },

    # ------------------------------------------------------------------------
    # 3) Asymmetric time-per-color
    # ------------------------------------------------------------------------
    @{
        Name = 'mcts_trad_white2s_black05s'
        Desc = 'MCTS_TRAD vs MCTS_TRAD, white 2s, black 0.5s — white advantage'
        Args = @('-w','MCTS_TRAD','-b','MCTS_TRAD','-m','B','-mtw','2','-mtb','0.5')
    },
    @{
        Name = 'mcts_trad_white05s_black2s'
        Desc = 'MCTS_TRAD vs MCTS_TRAD, white 0.5s, black 2s — black advantage'
        Args = @('-w','MCTS_TRAD','-b','MCTS_TRAD','-m','B','-mtw','0.5','-mtb','2')
    },
    @{
        Name = 'mcts_trad_white3s_black1s_ob'
        Desc = 'MCTS_TRAD vs MCTS_TRAD, white 3s, black 1s, opening book'
        Args = @('-w','MCTS_TRAD','-b','MCTS_TRAD','-m','B','-mtw','3','-mtb','1','-ob')
    },

    # ------------------------------------------------------------------------
    # 4) Debug mode (evaluator timing)
    # ------------------------------------------------------------------------
    @{
        Name = 'mcts_trad_selfplay_1s_debug'
        Desc = 'MCTS_TRAD vs MCTS_TRAD, 1s vs 1s, debug mode'
        Args = @('-w','MCTS_TRAD','-b','MCTS_TRAD','-m','B','-mtw','1','-mtb','1','-d')
    },

    # ------------------------------------------------------------------------
    # 5) MCTS_NN coverage (both equal and asymmetric)
    # ------------------------------------------------------------------------
    @{
        Name = 'mcts_nn_selfplay_1s'
        Desc = 'MCTS_NN vs MCTS_NN, 1s vs 1s'
        Args = @('-w','MCTS_NN','-b','MCTS_NN','-m','B','-mtw','1','-mtb','1')
    },
    @{
        Name = 'mcts_nn_selfplay_1s_ob'
        Desc = 'MCTS_NN vs MCTS_NN, 1s vs 1s, opening book enabled'
        Args = @('-w','MCTS_NN','-b','MCTS_NN','-m','B','-mtw','1','-mtb','1','-ob')
    },
    @{
        Name = 'mcts_nn_white2s_black05s'
        Desc = 'MCTS_NN vs MCTS_NN, white 2s, black 0.5s'
        Args = @('-w','MCTS_NN','-b','MCTS_NN','-m','B','-mtw','2','-mtb','0.5')
    },

    # ------------------------------------------------------------------------
    # 6) Cross-evaluator: MCTS_NN vs MCTS_TRAD (both color assignments)
    # ------------------------------------------------------------------------
    @{
        Name = 'mcts_nn_vs_mcts_trad_1s'
        Desc = 'MCTS_NN (W) vs MCTS_TRAD (B), 1s vs 1s'
        Args = @('-w','MCTS_NN','-b','MCTS_TRAD','-m','B','-mtw','1','-mtb','1')
    },
    @{
        Name = 'mcts_trad_vs_mcts_nn_1s'
        Desc = 'MCTS_TRAD (W) vs MCTS_NN (B), 1s vs 1s'
        Args = @('-w','MCTS_TRAD','-b','MCTS_NN','-m','B','-mtw','1','-mtb','1')
    },
    @{
        Name = 'mcts_nn_vs_mcts_trad_1s_ob'
        Desc = 'MCTS_NN (W) vs MCTS_TRAD (B), 1s vs 1s, opening book'
        Args = @('-w','MCTS_NN','-b','MCTS_TRAD','-m','B','-mtw','1','-mtb','1','-ob')
    },

    # ------------------------------------------------------------------------
    # 7) MCTS vs Minimax — both algorithms, both color assignments
    # ------------------------------------------------------------------------
    @{
        Name = 'mcts_trad_vs_minimax_trad_1s_d4'
        Desc = 'MCTS_TRAD (W, 1s) vs MINIMAX_TRAD (B, depth 4)'
        Args = @('-w','MCTS_TRAD','-b','MINIMAX_TRAD','-m','B','-mtw','1','-mtb','1','-db','4')
    },
    @{
        Name = 'minimax_trad_d4_vs_mcts_trad_1s'
        Desc = 'MINIMAX_TRAD (W, depth 4) vs MCTS_TRAD (B, 1s)'
        Args = @('-w','MINIMAX_TRAD','-b','MCTS_TRAD','-m','B','-dw','4','-mtw','1','-mtb','1')
    },
    @{
        Name = 'mcts_nn_vs_minimax_nn_1s_d3'
        Desc = 'MCTS_NN (W, 1s) vs MINIMAX_NN (B, depth 3)'
        Args = @('-w','MCTS_NN','-b','MINIMAX_NN','-m','B','-mtw','1','-mtb','1','-db','3')
    },
    @{
        Name = 'minimax_nn_d3_vs_mcts_nn_1s'
        Desc = 'MINIMAX_NN (W, depth 3) vs MCTS_NN (B, 1s)'
        Args = @('-w','MINIMAX_NN','-b','MCTS_NN','-m','B','-dw','3','-mtw','1','-mtb','1')
    },

    # ------------------------------------------------------------------------
    # 8) MCTS vs Stockfish — both color assignments, low Stockfish skill
    # ------------------------------------------------------------------------
    @{
        Name    = 'mcts_trad_vs_stockfish_1s'
        Desc    = 'MCTS_TRAD (W, 1s) vs STOCKFISH (B, depth 5, skill 3)'
        NeedsSF = $true
        Args    = @('-w','MCTS_TRAD','-b','STOCKFISH','-m','B','-mtw','1','-mtb','1','-dbs','5','-sb','3')
    },
    @{
        Name    = 'stockfish_vs_mcts_trad_1s'
        Desc    = 'STOCKFISH (W, depth 5, skill 3) vs MCTS_TRAD (B, 1s)'
        NeedsSF = $true
        Args    = @('-w','STOCKFISH','-b','MCTS_TRAD','-m','B','-dws','5','-sw','3','-mtw','1','-mtb','1')
    },
    @{
        Name    = 'mcts_nn_vs_stockfish_1s'
        Desc    = 'MCTS_NN (W, 1s) vs STOCKFISH (B, depth 5, skill 3)'
        NeedsSF = $true
        Args    = @('-w','MCTS_NN','-b','STOCKFISH','-m','B','-mtw','1','-mtb','1','-dbs','5','-sb','3')
    },
    @{
        Name    = 'stockfish_vs_mcts_nn_1s'
        Desc    = 'STOCKFISH (W, depth 5, skill 3) vs MCTS_NN (B, 1s)'
        NeedsSF = $true
        Args    = @('-w','STOCKFISH','-b','MCTS_NN','-m','B','-dws','5','-sw','3','-mtw','1','-mtb','1')
    },

    # ------------------------------------------------------------------------
    # 9) Custom start FEN (skipped if the FEN file is missing)
    # ------------------------------------------------------------------------
    @{
        Name     = 'mcts_trad_selfplay_1s_fen'
        Desc     = "MCTS_TRAD vs MCTS_TRAD, 1s vs 1s, start from out/$FenInput"
        NeedsFen = $true
        Args     = @('-w','MCTS_TRAD','-b','MCTS_TRAD','-m','B','-mtw','1','-mtb','1','-i',$FenInput)
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
$fenAvailable       = Test-Path -LiteralPath (Join-Path $outDir $FenInput)

# Filter scenarios by name pattern.
$selected = @($Scenarios | Where-Object { $_.Name -like $Filter })

if (-not $selected -or $selected.Count -eq 0) {
    Write-Error "No scenarios matched filter '$Filter'."
    exit 1
}

if ($ListOnly) {
    Write-Host '============================================================'
    Write-Host "Scenarios matching filter '$Filter':"
    Write-Host '============================================================'
    foreach ($s in $selected) {
        $flags = @()
        if ($s.NeedsSF)  { $flags += 'needs-stockfish' }
        if ($s.NeedsFen) { $flags += "needs-out/$FenInput" }
        if ($flags.Count) { $tag = '  [' + ($flags -join ', ') + ']' } else { $tag = '' }
        Write-Host ('  - {0,-40} {1}{2}' -f $s.Name, $s.Desc, $tag)
    }
    Write-Host ''
    Write-Host "Stockfish binary: $StockfishPath  (available: $stockfishAvailable)"
    Write-Host "FEN out\$FenInput available: $fenAvailable"
    exit 0
}

Write-Host '============================================================'
Write-Host "MCTS scenario batch — RunTag: $RunTag"
Write-Host "Filter: $Filter   Scenarios selected: $($selected.Count)"
Write-Host "Stockfish available: $stockfishAvailable  ($StockfishPath)"
Write-Host "FEN available:       $fenAvailable        (out\$FenInput)"
Write-Host '============================================================'

$results = New-Object System.Collections.Generic.List[object]
$index   = 0

foreach ($s in $selected) {
    $index++

    # Skip scenarios whose external prerequisites are missing.
    if ($s.NeedsSF -and -not $stockfishAvailable) {
        Write-Host ''
        Write-Host "[$index/$($selected.Count)] SKIP $($s.Name) — Stockfish binary not found."
        $results.Add([pscustomobject]@{ Name = $s.Name; Status = 'SKIPPED'; ExitCode = $null; Seconds = 0.0 })
        continue
    }
    if ($s.NeedsFen -and -not $fenAvailable) {
        Write-Host ''
        Write-Host "[$index/$($selected.Count)] SKIP $($s.Name) — FEN out\$FenInput not found."
        $results.Add([pscustomobject]@{ Name = $s.Name; Status = 'SKIPPED'; ExitCode = $null; Seconds = 0.0 })
        continue
    }

    # Build the final argv: copy scenario args, add stockfish path if needed,
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



