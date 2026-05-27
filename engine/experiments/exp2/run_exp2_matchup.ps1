<#
.SYNOPSIS
    Runs one matchup for Experiment 2 (Minimax depth scaling).

.DESCRIPTION
    Launch one instance per matchup in separate terminals for parallel execution.
    All matchups write to the same shared output directory so the final analysis
    sees all games together.

    Usage:
        .\run_exp2_matchup.ps1 -Matchup 1
        .\run_exp2_matchup.ps1 -Matchup 5 -GamesPerPair 30
        ...
        .\run_exp2_matchup.ps1 -Matchup 10

    Matchups:
        1  MINIMAX_TRAD d=2 vs MINIMAX_TRAD d=4
        2  MINIMAX_TRAD d=3 vs MINIMAX_TRAD d=4
        3  MINIMAX_TRAD d=4 vs MINIMAX_TRAD d=4  (sanity check)
        4  MINIMAX_TRAD d=5 vs MINIMAX_TRAD d=4
        5  MINIMAX_TRAD d=6 vs MINIMAX_TRAD d=4
        6  MINIMAX_NN   d=2 vs MINIMAX_NN   d=4
        7  MINIMAX_NN   d=3 vs MINIMAX_NN   d=4
        8  MINIMAX_NN   d=4 vs MINIMAX_NN   d=4  (sanity check)
        9  MINIMAX_NN   d=5 vs MINIMAX_NN   d=4
        10 MINIMAX_NN   d=6 vs MINIMAX_NN   d=4

    After all 10 finish, run:
        .\run_exp2_analyze.ps1
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

# Resolve engine/ directory relative to this script (experiments/exp2/ -> engine/)
$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $engineDir

if (-not (Test-Path -LiteralPath 'main.py')) {
    Write-Error "main.py not found in: $engineDir"
    exit 1
}

# ============================================================================
# MATCHUP DEFINITIONS — must match labels in exp2_minimax_depth.json
# ============================================================================

$matchups = @(
    @{ white='MINIMAX_TRAD'; black='MINIMAX_TRAD'; depth_white=2; depth_black=4; label='minimax_trad_d2_vs_minimax_trad_d4' },
    @{ white='MINIMAX_TRAD'; black='MINIMAX_TRAD'; depth_white=3; depth_black=4; label='minimax_trad_d3_vs_minimax_trad_d4' },
    @{ white='MINIMAX_TRAD'; black='MINIMAX_TRAD'; depth_white=4; depth_black=4; label='minimax_trad_d4_vs_minimax_trad_d4' },
    @{ white='MINIMAX_TRAD'; black='MINIMAX_TRAD'; depth_white=5; depth_black=4; label='minimax_trad_d5_vs_minimax_trad_d4' },
    @{ white='MINIMAX_TRAD'; black='MINIMAX_TRAD'; depth_white=6; depth_black=4; label='minimax_trad_d6_vs_minimax_trad_d4' },
    @{ white='MINIMAX_NN';   black='MINIMAX_NN';   depth_white=2; depth_black=4; label='minimax_nn_d2_vs_minimax_nn_d4' },
    @{ white='MINIMAX_NN';   black='MINIMAX_NN';   depth_white=3; depth_black=4; label='minimax_nn_d3_vs_minimax_nn_d4' },
    @{ white='MINIMAX_NN';   black='MINIMAX_NN';   depth_white=4; depth_black=4; label='minimax_nn_d4_vs_minimax_nn_d4' },
    @{ white='MINIMAX_NN';   black='MINIMAX_NN';   depth_white=5; depth_black=4; label='minimax_nn_d5_vs_minimax_nn_d4' },
    @{ white='MINIMAX_NN';   black='MINIMAX_NN';   depth_white=6; depth_black=4; label='minimax_nn_d6_vs_minimax_nn_d4' }
)

$matchupDef = $matchups[$Matchup - 1]
$label = $matchupDef.label

Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host "  EXP 2 — Matchup $Matchup/10" -ForegroundColor Green
Write-Host "  $($matchupDef.white) d=$($matchupDef.depth_white) vs $($matchupDef.black) d=$($matchupDef.depth_black)" -ForegroundColor Green
Write-Host "  Games: $GamesPerPair (swap colors)" -ForegroundColor Green
Write-Host "  Working dir: $engineDir" -ForegroundColor Green
Write-Host '================================================================' -ForegroundColor Green
Write-Host ''

# ============================================================================
# GENERATE SINGLE-MATCHUP CONFIG
# ============================================================================

$configPath = "experiments\exp2\_exp2_matchup${Matchup}.json"
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
$sharedDir = "exp2_minimax_depth_$ExperimentTag"

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
Write-Host "    $PSScriptRoot\run_exp2_analyze.ps1" -ForegroundColor Yellow

exit $exitCode
