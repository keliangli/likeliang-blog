param(
    [string]$RunAt = "08:40",
    [string]$TaskName = "keliangli-tech-news-agent"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runnerPath = Join-Path $scriptRoot "run.ps1"

if (-not (Test-Path $runnerPath)) {
    throw "未找到 run.ps1: $runnerPath"
}

$powerShellExe = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$runnerPath`" -AutoPush"

$action = New-ScheduledTaskAction -Execute $powerShellExe -Argument $arguments
$trigger = New-ScheduledTaskTrigger -Daily -At $RunAt
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null

Write-Host "已创建计划任务: $TaskName"
Write-Host "每日执行时间: $RunAt"
Write-Host "执行脚本: $runnerPath"

