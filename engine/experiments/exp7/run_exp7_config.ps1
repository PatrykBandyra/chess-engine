<#
.SYNOPSIS
    Runs one configuration of Experiment 7 (Opening book impact).

.DESCRIPTION
    4 configurations (book OFF / book ON x MINIMAX / MCTS, self-play):
        1  MINIMAX_TRAD d=4 -- book OFF
        2  MINIMAX_TRAD d=4 -- book ON
        3  MCTS_TRAD        -- book OFF
        4  MCTS_TRAD        -- book ON

    Launch 4 instances in separate terminals for parallel execution.

    Important: games start from STANDARD position (no openings file), because
    starting from an ECO opening would bypass the book entirely (the book is
    only consulted from positions covered in the book). The whole point of
    Exp 7 is to measure book impact on the opening phase, so games must start
    from move 1.

    Usage:
        .\run_exp7_config.ps1 -Config 1
        .\run_exp7_config.ps1 -Config 4 -GamesPerPair 40

    After all 4 finish, run:
        .\run_exp7_analyze.ps1

.NOTES
    For MCTS, the time budget defaults to 20s as specified in the research plan.
    Override with -McTsTime if running with lower budget for faster turnaround.
#>

param(
    [Parameter(Mandatory)]
    [ValidateRange(1, 4)]
    [int]$Config,

    [int]$GamesPerPair = 40,
    [int]$MinimaxDepth = 4,
    [double]$McTsTime = 20.0,
    [string]$ExperimentTag = '',
    [string]$StockfishPath = '..\stockfish_ai\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe',
    [switch]$Gui
)

$ErrorActionPreference = 'Stop'

# Resolve engine/ directory relative to this script
$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $engineDir

if (-not (Test-Path -LiteralPath 'main.py')) {
    Write-Error "main.py not found in: $engineDir"
    exit 1
}

# ============================================================================
# CONFIG DEFINITIONS
# ============================================================================

$configs = @(
    @{ name='minimax_trad_d4_book_off'; type='MINIMAX_TRAD';
       depth_white=$MinimaxDepth; depth_black=$MinimaxDepth; useBook=$false },
    @{ name='minimax_trad_d4_book_on';  type='MINIMAX_TRAD';
       depth_white=$MinimaxDepth; depth_black=$MinimaxDepth; useBook=$true },
    @{ name='mcts_trad_book_off';       type='MCTS_TRAD';
       mcts_time_white=$McTsTime; mcts_time_black=$McTsTime; useBook=$false },
    @{ name='mcts_trad_book_on';        type='MCTS_TRAD';
       mcts_time_white=$McTsTime; mcts_time_black=$McTsTime; useBook=$true }
)

$cfg = $configs[$Config - 1]
$configName = $cfg.name

# Build matchup definition (single self-play matchup per config)
$matchup = @{
    white = $cfg.type
    black = $cfg.type
    label = $configName
}
foreach ($key in @('depth_white', 'depth_black', 'mcts_time_white', 'mcts_time_black')) {
    if ($cfg.ContainsKey($key)) { $matchup[$key] = $cfg[$key] }
}

Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host "  EXP 7 -- Config $Config/4: $configName" -ForegroundColor Green
Write-Host "  Type: $($cfg.type) self-play" -ForegroundColor Green
Write-Host "  Book: $(if ($cfg.useBook) {'ON'} else {'OFF'})" -ForegroundColor Green
Write-Host "  Games: $GamesPerPair (swap colors)" -ForegroundColor Green
if ($cfg.ContainsKey('mcts_time_white')) {
    Write-Host "  MCTS time: $($cfg.mcts_time_white)s" -ForegroundColor Green
}
if ($cfg.ContainsKey('depth_white')) {
    Write-Host "  Minimax depth: $($cfg.depth_white)" -ForegroundColor Green
}
Write-Host "  Starting position: standard (no openings file)" -ForegroundColor Green
Write-Host '================================================================' -ForegroundColor Green
Write-Host ''

# ============================================================================
# GENERATE SINGLE-CONFIG JSON
# ============================================================================

$configPath = "experiments\exp7\_exp7_config${Config}.json"
$configJson = ConvertTo-Json @($matchup) -Depth 3
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
$sharedDir = "exp7_opening_book_$ExperimentTag"

# ============================================================================
# RUN -- note: NO -OpeningsFile (games from standard start)
# ============================================================================

$expArgs = @{
    ConfigFile = $configPath
    GamesPerPair = $GamesPerPair
    SwapColors = $true
    Adjudicate = $true
    StockfishPath = $StockfishPath
    OutputSubDir = $sharedDir
}
if ($cfg.useBook) { $expArgs.OpeningBook = $true }
if ($Gui) { $expArgs.Gui = $true }

& .\experiments\run_experiment.ps1 @expArgs
$exitCode = $LASTEXITCODE

Write-Host ''
if ($exitCode -eq 0) {
    Write-Host "  Config $Config ($configName) COMPLETE" -ForegroundColor Green
} else {
    Write-Host "  Config $Config ($configName) finished with errors (exit $exitCode)" -ForegroundColor Red
}
Write-Host "  Output: out\$sharedDir" -ForegroundColor Cyan
Write-Host ''
Write-Host '  When all 4 configs finish, run:' -ForegroundColor Yellow
Write-Host "    $PSScriptRoot\run_exp7_analyze.ps1" -ForegroundColor Yellow

exit $exitCode
