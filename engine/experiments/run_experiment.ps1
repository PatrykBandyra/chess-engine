<#
.SYNOPSIS
    Batch runner for chess engine experiments with JSON metrics collection.

.DESCRIPTION
    Reads an experiment configuration (JSON) and runs multiple games per matchup,
    optionally swapping colors and using fixed opening positions. Every game
    produces a JSONL metrics file via the -jl flag.

    Usage (run from engine/ folder):
        .\experiments\run_experiment.ps1 -ConfigFile experiments\exp1_round_robin.json
        .\experiments\run_experiment.ps1 -ConfigFile experiments\exp1_round_robin.json -GamesPerPair 50 -SwapColors -Adjudicate
        .\experiments\run_experiment.ps1 -ConfigFile experiments\exp1_round_robin.json -OpeningsFile experiments\openings_eco25.fen

    Config JSON format -- array of matchup objects:
    [
      {
        "white": "MCTS_TRAD",
        "black": "MINIMAX_TRAD",
        "label": "mcts_trad_vs_minimax_trad_d4",
        "mcts_time_white": 1.0,
        "depth_black": 4
      }
    ]

    Supported keys per matchup:
        white, black           (required) player type
        label                  (required) short identifier for filenames
        mcts_time_white/black  MCTS time budget (seconds)
        depth_white/black      Minimax depth
        depth_white_stockfish/depth_black_stockfish  Stockfish depth
        skill_white/skill_black  Stockfish skill level

.NOTES
    Run from the engine/ folder.
    Output: engine/out/<ConfigName>_<timestamp>/
#>

param(
    [Parameter(Mandatory)]
    [string]$ConfigFile,

    [int]$GamesPerPair = 50,
    [switch]$SwapColors,
    [switch]$Adjudicate,
    [double]$AdjudicateThreshold = 0.05,
    [int]$AdjudicateMoves = 20,
    [switch]$OpeningBook,
    [string]$OpeningsFile,
    [string]$StockfishPath = '..\stockfish_ai\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe',
    [string]$OutputSubDir,
    [switch]$Gui
)

$ErrorActionPreference = 'Stop'
$Python     = if ($IsMacOS -or $IsLinux) { 'python3' } else { 'python' }
$MainScript = 'main.py'

# ============================================================================
# VALIDATE
# ============================================================================

if (-not (Test-Path -LiteralPath $MainScript)) {
    Write-Error "File not found: $MainScript. Run this script from the 'engine' folder."
    exit 1
}

if (-not (Test-Path -LiteralPath $ConfigFile)) {
    Write-Error "Config file not found: $ConfigFile"
    exit 1
}

$config = Get-Content -Path $ConfigFile -Raw | ConvertFrom-Json

if ($config.Count -eq 0) {
    Write-Error "Config file is empty or invalid."
    exit 1
}

# Load openings (FEN strings, one per line, lines starting with # are comments)
$openings = @()
if ($OpeningsFile) {
    if (-not (Test-Path -LiteralPath $OpeningsFile)) {
        Write-Error "Openings file not found: $OpeningsFile"
        exit 1
    }
    $openings = Get-Content -Path $OpeningsFile |
        Where-Object { $_ -match '\S' -and $_ -notmatch '^\s*#' } |
        ForEach-Object { ($_ -split '\s*#')[0].Trim() }
    Write-Host "Loaded $($openings.Count) opening positions from $OpeningsFile"
}

# ============================================================================
# CREATE OUTPUT DIRECTORY
# ============================================================================

$configName = [System.IO.Path]::GetFileNameWithoutExtension($ConfigFile)
$runTag     = Get-Date -Format 'yyyyMMdd_HHmmss'
# engine.py prepends 'out/' to all file arguments (-g, -l, -jl, -i)
# so we pass paths relative to out/ as $outSubDir
if ($OutputSubDir) {
    $outSubDir = $OutputSubDir
} else {
    $outSubDir = "${configName}_${runTag}"
}
$outDir = Join-Path 'out' $outSubDir

New-Item -ItemType Directory -Path $outDir -Force | Out-Null
Write-Host "Output directory: $outDir" -ForegroundColor Cyan

# Copy config for reproducibility
Copy-Item -Path $ConfigFile -Destination (Join-Path $outDir '_config.json')

# ============================================================================
# HELPER: build main.py arguments for a single game
# ============================================================================

function Build-GameArgs {
    param(
        [object]$Matchup,
        [string]$White,
        [string]$Black,
        [string]$GameTag,
        [string]$FenFile
    )

    $args_ = @(
        '-w', $White,
        '-b', $Black,
        '-m', $(if ($Gui) { 'G' } else { 'B' }),
        '-g', (Join-Path $outSubDir "game_${GameTag}.txt"),
        '-l', (Join-Path $outSubDir "log_${GameTag}.txt"),
        '-jl', (Join-Path $outSubDir "metrics_${GameTag}.jsonl")
    )

    # MCTS time budgets -- respect swap
    $mtw = if ($White -eq $Matchup.white) { $Matchup.mcts_time_white } else { $Matchup.mcts_time_black }
    $mtb = if ($Black -eq $Matchup.black) { $Matchup.mcts_time_black } else { $Matchup.mcts_time_white }
    if ($mtw) { $args_ += @('-mtw', "$mtw") }
    if ($mtb) { $args_ += @('-mtb', "$mtb") }

    # Minimax depths -- respect swap
    $dw = if ($White -eq $Matchup.white) { $Matchup.depth_white } else { $Matchup.depth_black }
    $db = if ($Black -eq $Matchup.black) { $Matchup.depth_black } else { $Matchup.depth_white }
    if ($dw) { $args_ += @('-dw', "$dw") }
    if ($db) { $args_ += @('-db', "$db") }

    # Stockfish depths -- respect swap
    $dws = if ($White -eq $Matchup.white) { $Matchup.depth_white_stockfish } else { $Matchup.depth_black_stockfish }
    $dbs = if ($Black -eq $Matchup.black) { $Matchup.depth_black_stockfish } else { $Matchup.depth_white_stockfish }
    if ($dws) { $args_ += @('-dws', "$dws") }
    if ($dbs) { $args_ += @('-dbs', "$dbs") }

    # Stockfish skill -- respect swap
    $sw = if ($White -eq $Matchup.white) { $Matchup.skill_white } else { $Matchup.skill_black }
    $sb = if ($Black -eq $Matchup.black) { $Matchup.skill_black } else { $Matchup.skill_white }
    if ($sw) { $args_ += @('-sw', "$sw") }
    if ($sb) { $args_ += @('-sb', "$sb") }

    # Stockfish path (if either player is Stockfish)
    if ($White -eq 'STOCKFISH' -or $Black -eq 'STOCKFISH') {
        $args_ += @('-sp', $StockfishPath)
    }

    # Opening position
    if ($FenFile) {
        $args_ += @('-i', $FenFile)
    }

    # Opening book
    if ($OpeningBook) {
        $args_ += '-ob'
    }

    # Adjudication
    if ($Adjudicate) {
        $args_ += @('-adj', '-adjt', "$AdjudicateThreshold", '-adjm', "$AdjudicateMoves")
    }

    return $args_
}

# ============================================================================
# RUN GAMES
# ============================================================================

$totalGames   = 0
$completedOk  = 0
$completedErr = 0
$startTime    = Get-Date

$resultsLog = @()

Write-Host ''
Write-Host ('=' * 70) -ForegroundColor Green
Write-Host "  Experiment: $configName"
Write-Host "  Matchups:   $($config.Count)"
Write-Host "  Games/pair: $GamesPerPair $(if ($SwapColors) {'(swap colors)'} else {'(no swap)'})"
Write-Host "  Openings:   $(if ($openings.Count -gt 0) {"$($openings.Count) positions"} else {'standard start'})"
Write-Host "  Adjudicate: $(if ($Adjudicate) {"yes (threshold=$AdjudicateThreshold, moves=$AdjudicateMoves)"} else {'no'})"
Write-Host "  Mode:       $(if ($Gui) {'GUI (graphical, slower, interactive)'} else {'Background (headless)'})"
Write-Host ('=' * 70) -ForegroundColor Green
if ($Gui) {
    Write-Host '  WARNING: GUI mode runs each game in a window -- close it to advance.' -ForegroundColor Yellow
    Write-Host '  Best for debugging / demos. For batch runs use background mode.' -ForegroundColor Yellow
    Write-Host ''
}
Write-Host ''

foreach ($matchup in $config) {
    $label = $matchup.label
    Write-Host "--- Matchup: $label ($($matchup.white) vs $($matchup.black)) ---" -ForegroundColor Yellow

    for ($gameIdx = 1; $gameIdx -le $GamesPerPair; $gameIdx++) {

        # Determine color assignment
        $swapped = $false
        if ($SwapColors -and $gameIdx -gt [math]::Ceiling($GamesPerPair / 2)) {
            $swapped = $true
        }

        $white = if ($swapped) { $matchup.black } else { $matchup.white }
        $black = if ($swapped) { $matchup.white } else { $matchup.black }
        $colorTag = if ($swapped) { 'swap' } else { 'orig' }

        # Determine opening position
        $fenTempFile = $null
        if ($openings.Count -gt 0) {
            $openingIdx = ($gameIdx - 1) % $openings.Count
            $fen = $openings[$openingIdx]
            $fenTempName = "_temp_opening_${label}_g${gameIdx}.fen"
            $fenTempFile = $fenTempName
            # Write FEN without BOM (PowerShell 5.1's -Encoding utf8 adds BOM, which python-chess rejects)
            [System.IO.File]::WriteAllText(
                (Join-Path (Get-Location) 'out' $fenTempName),
                $fen,
                [System.Text.UTF8Encoding]::new($false)
            )
        }

        $gameTag = "${label}_g${gameIdx}_${colorTag}"
        $gameArgs = Build-GameArgs -Matchup $matchup -White $white -Black $black -GameTag $gameTag -FenFile $fenTempFile

        $totalGames++
        $pct = [math]::Round(($totalGames - 1) / ($config.Count * $GamesPerPair) * 100, 0)
        Write-Host "  [$pct%] Game $gameIdx/$GamesPerPair ($white vs $black) ... " -NoNewline

        $gameStart = Get-Date
        & $Python $MainScript @gameArgs
        $exitCode = $LASTEXITCODE
        $elapsed = ((Get-Date) - $gameStart).TotalSeconds

        # Read result from metrics file
        $metricsPath = Join-Path $outDir "metrics_${gameTag}.jsonl"
        $result = '?'
        $termination = '?'
        if (Test-Path -LiteralPath $metricsPath) {
            $lastLine = Get-Content $metricsPath -Tail 1
            try {
                $summary = $lastLine | ConvertFrom-Json
                if ($summary.type -eq 'game_summary') {
                    $result = $summary.result
                    $termination = $summary.termination
                }
            } catch {}
        }

        if ($exitCode -eq 0) {
            $completedOk++
            Write-Host "$result ($termination) in $([math]::Round($elapsed, 1))s" -ForegroundColor Green
        } else {
            $completedErr++
            Write-Host "ERROR (exit $exitCode) in $([math]::Round($elapsed, 1))s" -ForegroundColor Red
        }

        $resultsLog += [PSCustomObject]@{
            Matchup     = $label
            Game        = $gameIdx
            White       = $white
            Black       = $black
            Swapped     = $swapped
            OpeningIdx  = if ($openings.Count -gt 0) { ($gameIdx - 1) % $openings.Count } else { -1 }
            Result      = $result
            Termination = $termination
            TimeSec     = [math]::Round($elapsed, 2)
            ExitCode    = $exitCode
        }

        # Clean up temp FEN file
        $fenTempPath = if ($fenTempFile) { Join-Path 'out' $fenTempFile } else { $null }
        if ($fenTempFile -and (Test-Path -LiteralPath $fenTempPath)) {
            Remove-Item $fenTempPath -Confirm:$false
        }
    }
    Write-Host ''
}

# ============================================================================
# SUMMARY
# ============================================================================

$totalElapsed = ((Get-Date) - $startTime).TotalSeconds

Write-Host ('=' * 70) -ForegroundColor Cyan
Write-Host "  EXPERIMENT COMPLETE: $configName" -ForegroundColor Cyan
Write-Host "  Total games: $totalGames (OK: $completedOk, Errors: $completedErr)"
Write-Host "  Total time:  $([math]::Round($totalElapsed / 60, 1)) minutes"
Write-Host "  Output:      $outDir"
Write-Host ('=' * 70) -ForegroundColor Cyan

# Save results summary CSV
$csvPath = Join-Path $outDir '_results.csv'
$resultsLog | Export-Csv -Path $csvPath -NoTypeInformation -Encoding utf8
Write-Host "Results CSV: $csvPath"

# Print W/D/L per matchup
Write-Host ''
Write-Host 'Results per matchup:' -ForegroundColor Yellow
Write-Host ('-' * 60)

$grouped = $resultsLog | Group-Object -Property Matchup
foreach ($group in $grouped) {
    $wins   = ($group.Group | Where-Object { $_.Result -eq '1-0' }).Count
    $draws  = ($group.Group | Where-Object { $_.Result -eq '1/2-1/2' }).Count
    $losses = ($group.Group | Where-Object { $_.Result -eq '0-1' }).Count
    $errors = ($group.Group | Where-Object { $_.ExitCode -ne 0 }).Count
    Write-Host ("  {0,-40} W:{1} D:{2} L:{3} E:{4}" -f $group.Name, $wins, $draws, $losses, $errors)
}
Write-Host ('-' * 60)

exit $(if ($completedErr -gt 0) { 1 } else { 0 })
