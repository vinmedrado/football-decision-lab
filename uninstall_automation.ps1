$ErrorActionPreference = "Stop"

foreach ($TaskName in @(
    "FootballDecisionLab-Paper-Automation",
    "FootballDecisionLab-Dashboard",
    "FootballLab-Paper-Automation",
    "FootballLab-Dashboard",
    "FootballLab_Daily_Update"
)) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Tarefa removida: $TaskName"
    }
    else {
        Write-Host "Tarefa não encontrada: $TaskName"
    }
}

$RunKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
foreach ($RunName in @("FootballDecisionLabDashboard", "FootballLabDashboard")) {
    if (Get-ItemProperty -Path $RunKey -Name $RunName -ErrorAction SilentlyContinue) {
        Remove-ItemProperty -Path $RunKey -Name $RunName
        Write-Host "Inicialização removida: $RunName"
    }
}
