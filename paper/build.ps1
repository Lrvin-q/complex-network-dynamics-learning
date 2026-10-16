param(
    [switch]$RebuildFigures,
    [switch]$Clean,
    [string]$TexLiveBin = $env:TEXLIVE_BIN,
    [string]$PythonPath = $env:PAPER_PYTHON
)
$ErrorActionPreference = 'Stop'

$PaperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $PaperDir
. (Join-Path $PaperDir 'scripts\resolve_environment.ps1')
$TexLiveBin = Find-PaperTexLive -RequestedPath $TexLiveBin
$ExperimentPython = Find-PaperPython -RequestedPath $PythonPath -ProjectRoot $RootDir

$env:PATH = "$TexLiveBin;$env:PATH"

if ($RebuildFigures) {
    & $ExperimentPython (Join-Path $PaperDir 'scripts\build_figures.py')
    if ($LASTEXITCODE -ne 0) { throw 'Notebook figure generation failed' }
}
& $ExperimentPython (Join-Path $PaperDir 'scripts\verify_results.py')
if ($LASTEXITCODE -ne 0) { throw 'Result verification failed' }

$BuildDir = Join-Path $PaperDir 'build'
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
if ($Clean) {
    $ResolvedBuild = [IO.Path]::GetFullPath($BuildDir)
    $ExpectedBuild = [IO.Path]::GetFullPath((Join-Path $PaperDir 'build'))
    if ($ResolvedBuild -ne $ExpectedBuild) { throw 'Unsafe build path' }
    Get-ChildItem -LiteralPath $ResolvedBuild -File | Remove-Item -Force
}

Push-Location $PaperDir
try {
    & (Join-Path $TexLiveBin 'latexmk.exe') -xelatex -synctex=1 -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
    if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed' }
    $LogText = Get-Content -Raw -LiteralPath (Join-Path $BuildDir 'main.log')
    if ($LogText -match 'Overfull \\[hv]box|Missing character:|undefined references|Citation .+ undefined|LaTeX Font Warning') {
        throw 'PDF validation failed. Inspect paper/build/main.log.'
    }
}
finally {
    Pop-Location
}

$OutputDir = Join-Path $RootDir 'output\pdf'
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputName = "$([char]0x590d)$([char]0x6742)$([char]0x7f51)$([char]0x7edc)$([char]0x52a8)$([char]0x529b)$([char]0x5b66)$([char]0x5b66)$([char]0x4e60)$([char]0x4e0e)$([char]0x62d3)$([char]0x6251)$([char]0x91cd)$([char]0x6784).pdf"
$OutputPdf = Join-Path $OutputDir $OutputName
Copy-Item -LiteralPath (Join-Path $BuildDir 'main.pdf') -Destination $OutputPdf -Force
Write-Host "Built: $OutputPdf"
