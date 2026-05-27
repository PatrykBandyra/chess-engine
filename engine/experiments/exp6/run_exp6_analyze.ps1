<#
.SYNOPSIS
    Aggregates Experiment 6 results from all 4 variant CSV files.

.DESCRIPTION
    Reads exp6_variant<N>_<name>_<tag>.csv files in this directory and produces:
        - exp6_solve_rate.csv             -- solve rate per variant
        - exp6_solve_by_theme.csv         -- solve rate per (variant, theme)
        - exp6_minimax_depth_to_solve.csv -- min depth stats for Minimax
        - exp6_summary.txt                -- human-readable
        - plots/exp6_solve_rate_bars.png
        - plots/exp6_solve_by_theme_heatmap.png
        - plots/exp6_solve_by_theme_radar.png (if reasonable # of themes)
        - plots/exp6_minimax_depth_hist.png

    Usage:
        .\run_exp6_analyze.ps1
        .\run_exp6_analyze.ps1 -Tag 20260527
#>

param(
    [string]$Tag = '',
    [string]$Python = ''
)

if (-not $Python) { $Python = if ($IsMacOS -or $IsLinux) { 'python3' } else { 'python' } }

$engineDir = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $engineDir

$AnalysisScript = Join-Path $PSScriptRoot 'exp6_analyze.py'

if (-not (Test-Path -LiteralPath $AnalysisScript)) {
    Write-Error "Analysis script not found: $AnalysisScript"
    exit 1
}

# Count input CSVs
$pattern = if ($Tag) { "exp6_variant*_${Tag}.csv" } else { 'exp6_variant*.csv' }
$inputs = @(Get-ChildItem -Path $PSScriptRoot -Filter $pattern -ErrorAction SilentlyContinue)

Write-Host ''
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host '  EXP 6 -- Tactical accuracy analysis' -ForegroundColor Cyan
Write-Host "  Input files: $($inputs.Count)" -ForegroundColor Cyan
if ($Tag) { Write-Host "  Tag filter: $Tag" -ForegroundColor Cyan }
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host ''

if ($inputs.Count -eq 0) {
    Write-Error "No exp6_variant*.csv files found in $PSScriptRoot. Run run_exp6_variant.ps1 first."
    exit 1
}

$pyArgs = @($AnalysisScript)
if ($Tag) { $pyArgs += @('--tag', $Tag) }

& $Python @pyArgs
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host ''
    Write-Host '================================================================' -ForegroundColor Green
    Write-Host '  EXP 6 ANALYSIS COMPLETE' -ForegroundColor Green
    Write-Host "  Results in: $PSScriptRoot" -ForegroundColor Green
    Write-Host '  Key files:' -ForegroundColor Green
    Write-Host '    exp6_solve_rate.csv             -- solve rate per variant' -ForegroundColor Green
    Write-Host '    exp6_solve_by_theme.csv         -- solve rate per theme' -ForegroundColor Green
    Write-Host '    exp6_minimax_depth_to_solve.csv -- depth stats (Minimax)' -ForegroundColor Green
    Write-Host '    exp6_summary.txt                -- human-readable' -ForegroundColor Green
    Write-Host '    plots\exp6_solve_rate_bars.png' -ForegroundColor Green
    Write-Host '    plots\exp6_solve_by_theme_heatmap.png' -ForegroundColor Green
    Write-Host '    plots\exp6_solve_by_theme_radar.png' -ForegroundColor Green
    Write-Host '    plots\exp6_minimax_depth_hist.png' -ForegroundColor Green
    Write-Host '================================================================' -ForegroundColor Green
}

exit $exitCode
