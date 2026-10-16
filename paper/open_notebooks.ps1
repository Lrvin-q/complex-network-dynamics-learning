param([string]$PythonPath = $env:PAPER_PYTHON)
$ErrorActionPreference = 'Stop'
$PaperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $PaperDir 'scripts\resolve_environment.ps1')
$NotebookPython = Find-PaperPython -RequestedPath $PythonPath -ProjectRoot (Split-Path -Parent $PaperDir)
$env:PATH = "$(Split-Path -Parent $NotebookPython);$env:PATH"
& $NotebookPython -m jupyterlab "--notebook-dir=$(Join-Path $PaperDir 'notebooks')"
exit $LASTEXITCODE
