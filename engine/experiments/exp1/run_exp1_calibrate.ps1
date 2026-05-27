<#
.SYNOPSIS
    Calibration for Experiment 1: measures Minimax d=4 avg time per move.

.DESCRIPTION
    Runs a few short MINIMAX_TRAD d=4 vs d=4 games, parses metrics to get
    average time per move, saves it to experiments\exp1\_mcts_calibrated_time.txt.
    Run this ONCE before launching the 6 parallel pair scripts.

    Usage (from any directory):
        .\run_exp1_calibrate.ps1
        .\run_exp1_calibrate.ps1 -CalibrationGames 6 -Depth 5
#>

param(
    [int]$CalibrationGames = 3,
    [int]$Depth = 4
)

$ErrorActionPreference = 'Stop'
$Python     = 'python'

# Resolve engine/ directory relative to this script
$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
$MainScript = Join-Path $engineDir 'main.py'

if (-not (Test-Path -LiteralPath $MainScript)) {
    Write-Error "main.py not found at: $MainScript"
    exit 1
}

Set-Location $engineDir

$outDir = 'out'
if (-not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$calTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$calSubDir = "exp1_calibration_$calTag"

New-Item -ItemType Directory -Path "$outDir\$calSubDir" -Force | Out-Null

Write-Host ''
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host '  CALIBRATION: Measuring MINIMAX_TRAD d='$Depth' avg time/move' -ForegroundColor Cyan
Write-Host '  Games: '$CalibrationGames -ForegroundColor Cyan
Write-Host '  Working dir: '$engineDir -ForegroundColor Cyan
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host ''

for ($i = 1; $i -le $CalibrationGames; $i++) {
    $calArgs = @(
        '-w', 'MINIMAX_TRAD', '-b', 'MINIMAX_TRAD',
        '-m', 'B',
        '-dw', "$Depth", '-db', "$Depth",
        '-adj',
        '-g', "$calSubDir\cal_game_$i.txt",
        '-l', "$calSubDir\cal_log_$i.txt",
        '-jl', "$calSubDir\cal_metrics_$i.jsonl"
    )
    Write-Host "  Calibration game $i/$CalibrationGames ... " -NoNewline
    $t0 = Get-Date
    & $Python $MainScript @calArgs
    $elapsed = ((Get-Date) - $t0).TotalSeconds
    Write-Host "done ($([math]::Round($elapsed, 1))s)"
}

# Parse JSONL files to compute avg time per move
$totalTime = 0.0
$totalMoves = 0
foreach ($f in (Get-ChildItem "$outDir\$calSubDir\cal_metrics_*.jsonl")) {
    foreach ($line in (Get-Content $f.FullName -Encoding utf8)) {
        if (-not $line.Trim()) { continue }
        try {
            $obj = $line | ConvertFrom-Json
            if ($obj.type -eq 'game_summary') { continue }
            if ($null -ne $obj.time_s) {
                $totalTime += $obj.time_s
                $totalMoves++
            }
        } catch {}
    }
}

if ($totalMoves -gt 0) {
    $avgTime = [math]::Round($totalTime / $totalMoves, 3)
} else {
    Write-Warning "Could not measure avg time. Defaulting to 1.0s."
    $avgTime = 1.0
}

# Save calibrated time next to this script
$calibFile = Join-Path $PSScriptRoot '_mcts_calibrated_time.txt'
[System.IO.File]::WriteAllText($calibFile, "$avgTime", [System.Text.UTF8Encoding]::new($false))

Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host "  Calibration complete: $totalMoves moves measured" -ForegroundColor Green
Write-Host "  Average time per move: $avgTime s" -ForegroundColor Green
Write-Host "  Saved to: $calibFile" -ForegroundColor Green
Write-Host '================================================================' -ForegroundColor Green
Write-Host ''
Write-Host 'Now launch 6 pair scripts in parallel:' -ForegroundColor Yellow
for ($p = 1; $p -le 6; $p++) {
    Write-Host "  $PSScriptRoot\run_exp1_pair.ps1 -Pair $p" -ForegroundColor Yellow
}
