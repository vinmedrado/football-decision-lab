$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonCommand = Get-Command python -ErrorAction Stop
$PythonExe = $PythonCommand.Source
$PythonwExe = Join-Path (Split-Path -Parent $PythonExe) "pythonw.exe"
if (-not (Test-Path -LiteralPath $PythonwExe -PathType Leaf)) {
    throw "pythonw.exe não encontrado ao lado de: $PythonExe"
}
$HeadlessLauncher = Join-Path $ProjectRoot "scripts\run_headless.py"
if (-not (Test-Path -LiteralPath $HeadlessLauncher -PathType Leaf)) {
    throw "Inicializador silencioso não encontrado: $HeadlessLauncher"
}
$Orchestrator = Join-Path $ProjectRoot "automation_orchestrator.py"
$Dashboard = Join-Path $ProjectRoot "web_dashboard_lux\app.py"

$AutomationTask = "FootballDecisionLab-Paper-Automation"
$DashboardTask = "FootballDecisionLab-Dashboard"
$AutomationAction = New-ScheduledTaskAction `
    -Execute $PythonwExe `
    -Argument "`"$HeadlessLauncher`" automation `"$Orchestrator`" --run-due" `
    -WorkingDirectory $ProjectRoot
$AutomationTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 15) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$AutomationSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 12)
Register-ScheduledTask `
    -TaskName $AutomationTask `
    -Action $AutomationAction `
    -Trigger $AutomationTrigger `
    -Settings $AutomationSettings `
    -Description "Football Decision Lab: controlador paper idempotente; apostas reais bloqueadas." `
    -Force | Out-Null

$DashboardAction = New-ScheduledTaskAction `
    -Execute $PythonwExe `
    -Argument "`"$HeadlessLauncher`" dashboard `"$Dashboard`"" `
    -WorkingDirectory $ProjectRoot
$DashboardTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$DashboardSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero)
$DashboardStartupMode = "tarefa agendada"
try {
    Register-ScheduledTask `
        -TaskName $DashboardTask `
        -Action $DashboardAction `
        -Trigger $DashboardTrigger `
        -Settings $DashboardSettings `
        -Description "Football Decision Lab: dashboard local na porta 8060." `
        -Force | Out-Null
}
catch {
    $RunKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
    New-Item -Path $RunKey -Force | Out-Null
    $DashboardCommand = "`"$PythonwExe`" `"$Dashboard`""
    Set-ItemProperty -Path $RunKey -Name "FootballDecisionLabDashboard" -Value $DashboardCommand
    $DashboardStartupMode = "inicialização do usuário"
}

Write-Host ""
Write-Host "Automação instalada:"
Write-Host "  $AutomationTask - verificação silenciosa a cada 15 minutos"
Write-Host "  Dashboard - mantido ativo em segundo plano via $DashboardStartupMode"
Write-Host ""
Write-Host "O controlador faz recuperação automática de horários perdidos quando o notebook volta a ligar."
Write-Host "Nenhuma janela de terminal precisa permanecer aberta."
