<#
.SYNOPSIS
    Runs one matchup for Experiment 3 (MCTS time scaling).

.DESCRIPTION
    Launch one instance per matchup in separate terminals for parallel execution.
    All matchups write to the same shared output directory so the final analysis
    sees all games together.

    Usage:
        .\run_exp3_matchup.ps1 -Matchup 1
        .\run_exp3_matchup.ps1 -Matchup 5 -GamesPerPair 30
        ...
        .\run_exp3_matchup.ps1 -Matchup 10

    Matchups:
        1  MCTS_TRAD 1s   vs MCTS_TRAD 20s
        2  MCTS_TRAD 5s   vs MCTS_TRAD 20s
        3  MCTS_TRAD 10s  vs MCTS_TRAD 20s
        4  MCTS_TRAD 20s  vs MCTS_TRAD 20s  (sanity check)
        5  MCTS_TRAD 40s  vs MCTS_TRAD 20s
        6  MCTS_NN   1s   vs MCTS_NN   20s
        7  MCTS_NN   5s   vs MCTS_NN   20s
        8  MCTS_NN   10s  vs MCTS_NN   20s
        9  MCTS_NN   20s  vs MCTS_NN   20s  (sanity check)
        10 MCTS_NN   40s  vs MCTS_NN   20s

    WARNING: matchups 5 and 10 (40s vs 20s) are heavy — ~40+h per matchup at 30 games.
    Consider lowering -GamesPerPair for those if time-constrained.

    After all 10 finish, run:
        .\run_exp3_analyze.ps1
#>

param(
    [Parameter(Mandatory)]
    [ValidateRange(1, 10)]
    [int]$Matchup,

    [int]$GamesPerPair = 30,
    [string]$ExperimentTag = '',
    [switch]$Gui
)

$ErrorActionPreference = 'Stop'

# Resolve engine/ directory relative to this script (experiments/exp3/ -> engine/)
$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $engineDir

if (-not (Test-Path -LiteralPath 'main.py')) {
    Write-Error "main.py not found in: $engineDir"
    exit 1
}

# ============================================================================
# MATCHUP DEFINITIONS — labels must match exp3_mcts_time.json
# ============================================================================

$matchups = @(
    @{ white='MCTS_TRAD'; black='MCTS_TRAD'; mcts_time_white=1.0;  mcts_time_black=20.0; label='mcts_trad_1s_vs_mcts_trad_20s' },
    @{ white='MCTS_TRAD'; black='MCTS_TRAD'; mcts_time_white=5.0;  mcts_time_black=20.0; label='mcts_trad_5s_vs_mcts_trad_20s' },
    @{ white='MCTS_TRAD'; black='MCTS_TRAD'; mcts_time_white=10.0; mcts_time_black=20.0; label='mcts_trad_10s_vs_mcts_trad_20s' },
    @{ white='MCTS_TRAD'; black='MCTS_TRAD'; mcts_time_white=20.0; mcts_time_black=20.0; label='mcts_trad_20s_vs_mcts_trad_20s' },
    @{ white='MCTS_TRAD'; black='MCTS_TRAD'; mcts_time_white=40.0; mcts_time_black=20.0; label='mcts_trad_40s_vs_mcts_trad_20s' },
    @{ white='MCTS_NN';   black='MCTS_NN';   mcts_time_white=1.0;  mcts_time_black=20.0; label='mcts_nn_1s_vs_mcts_nn_20s' },
    @{ white='MCTS_NN';   black='MCTS_NN';   mcts_time_white=5.0;  mcts_time_black=20.0; label='mcts_nn_5s_vs_mcts_nn_20s' },
    @{ white='MCTS_NN';   black='MCTS_NN';   mcts_time_white=10.0; mcts_time_black=20.0; label='mcts_nn_10s_vs_mcts_nn_20s' },
    @{ white='MCTS_NN';   black='MCTS_NN';   mcts_time_white=20.0; mcts_time_black=20.0; label='mcts_nn_20s_vs_mcts_nn_20s' },
    @{ white='MCTS_NN';   black='MCTS_NN';   mcts_time_white=40.0; mcts_time_black=20.0; label='mcts_nn_40s_vs_mcts_nn_20s' }
)

$matchupDef = $matchups[$Matchup - 1]
$label = $matchupDef.label

Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host "  EXP 3 — Matchup $Matchup/10" -ForegroundColor Green
Write-Host "  $($matchupDef.white) t=$($matchupDef.mcts_time_white)s vs $($matchupDef.black) t=$($matchupDef.mcts_time_black)s" -ForegroundColor Green
Write-Host "  Games: $GamesPerPair (swap colors)" -ForegroundColor Green
Write-Host "  Working dir: $engineDir" -ForegroundColor Green
Write-Host '================================================================' -ForegroundColor Green
Write-Host ''

# ============================================================================
# GENERATE SINGLE-MATCHUP CONFIG
# ============================================================================

$configPath = "experiments\exp3\_exp3_matchup${Matchup}.json"
$configJson = ConvertTo-Json @($matchupDef) -Depth 3
[System.IO.File]::WriteAllText(
    (Join-Path $engineDir $configPath),
    $configJson,
    [System.Text.UTF8Encoding]::new($false)
)

# ============================================================================
# SHARED OUTPUT DIRECTORY (per day)
# ============================================================================

if (-not $ExperimentTag) {
    $ExperimentTag = Get-Date -Format 'yyyyMMdd'
}
$sharedDir = "exp3_mcts_time_$ExperimentTag"

# ============================================================================
# RUN
# ============================================================================

$expArgs = @(
    '-ConfigFile', $configPath,
    '-GamesPerPair', $GamesPerPair,
    '-SwapColors',
    '-Adjudicate',
    '-OpeningsFile', 'experiments\openings_eco25.fen',
    '-OutputSubDir', $sharedDir
)

if ($Gui) { $expArgs += '-Gui' }

& .\experiments\run_experiment.ps1 @expArgs
$exitCode = $LASTEXITCODE

Write-Host ''
if ($exitCode -eq 0) {
    Write-Host "  Matchup $Matchup ($label) COMPLETE" -ForegroundColor Green
} else {
    Write-Host "  Matchup $Matchup ($label) finished with errors (exit $exitCode)" -ForegroundColor Red
}
Write-Host "  Output: out\$sharedDir" -ForegroundColor Cyan
Write-Host ''
Write-Host '  When all 10 matchups finish, run:' -ForegroundColor Yellow
Write-Host "    $PSScriptRoot\run_exp3_analyze.ps1" -ForegroundColor Yellow

exit $exitCode
