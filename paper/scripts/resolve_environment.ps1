# Shared Windows runtime resolution. Explicit overrides always take precedence.
function Find-PaperPython {
    param([string]$RequestedPath, [string]$ProjectRoot)
    if ($RequestedPath) {
        if (-not (Test-Path -LiteralPath $RequestedPath -PathType Leaf)) {
            throw "Configured Python not found: $RequestedPath"
        }
        return (Get-Item -LiteralPath $RequestedPath).FullName
    }
    $PythonCandidates = @(
        (Join-Path $ProjectRoot '.venv\Scripts\python.exe'),
        'D:\L\Document\Code\Python\PythonFile\.venv\Scripts\python.exe'
    )
    foreach ($Candidate in $PythonCandidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) { return $Candidate }
    }
    foreach ($CommandName in @('python.exe', 'python3.exe')) {
        $PythonCommand = Get-Command $CommandName -ErrorAction SilentlyContinue
        if ($PythonCommand) { return $PythonCommand.Source }
    }
    throw 'Python not found. Set PAPER_PYTHON or create a project .venv.'
}

function Find-PaperTexLive {
    param([string]$RequestedPath)
    $TexCandidates = @()
    if ($RequestedPath) {
        $TexCandidates = @($RequestedPath)
    } else {
        $TexCandidates += 'D:\SelfDownloadSoftware\texlive\2025\bin\windows'
        $XeCommand = Get-Command xelatex.exe -ErrorAction SilentlyContinue
        if ($XeCommand) { $TexCandidates += Split-Path -Parent $XeCommand.Source }
    }
    foreach ($Candidate in $TexCandidates) {
        $Complete = $true
        foreach ($ToolName in @('xelatex.exe', 'latexmk.exe', 'biber.exe')) {
            if (-not (Test-Path -LiteralPath (Join-Path $Candidate $ToolName) -PathType Leaf)) {
                $Complete = $false
            }
        }
        if ($Complete) { return (Get-Item -LiteralPath $Candidate).FullName }
    }
    throw 'LaTeX tools not found. Set TEXLIVE_BIN to the directory containing xelatex, latexmk and biber.'
}
