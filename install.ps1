#Set-ExecutionPolicy Bypass -Scope Process -Force
#.\install.ps1
#run htese lines to allow the script to run

# Get the directory of the current script
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
# Path to the launch.ps1 script
$launchScriptPath = Join-Path $scriptDir "launch.ps1"
$startupFolder = [System.Environment]::GetFolderPath("Startup")

# Path to the Startup folder
$shortcutPath = "$startupFolder\LaunchPasswordManager.lnk"

# install all dependencies
pip install -r requirements_windows.txt

Write-Host "This script will register the install.ps1 script to run at startup."
$consent = Read-Host "Do you want to continue? (y/n)"

if ($consent -eq "y" -or $consent -eq "Y") {
    # Create a WScript.Shell COM object
    $shell = New-Object -ComObject WScript.Shell

    # Create a shortcut
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = "powershell.exe"
    $shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$launchScriptPath`""
    $shortcut.WorkingDirectory = (Get-Location).Path
    $shortcut.Save()

    Write-Host "The launch script has been registered to run at startup."
} else {
    Write-Host "Installation aborted."
}
