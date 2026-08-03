$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
& python (Join-Path $ProjectRoot "automation_orchestrator.py") --run-due
exit $LASTEXITCODE
