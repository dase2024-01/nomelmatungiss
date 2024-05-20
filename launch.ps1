#move to the folder


cd ~/PycharmProjects/pythonProject/

./Scripts/activate

# Get the system locale
$systemLocale = Get-WinSystemLocale

# Get the user locale
$userLocale = Get-Culture

# Print the locales to verify
Write-Host "System Locale: $($systemLocale.Name)"
Write-Host "User Locale: $($userLocale.Name)"

# Define the port number
$port = 8000

# Find the process ID (PID) of the process using the specified port
$pids = Get-NetTCPConnection -LocalPort $port -State Listen | Select-Object -ExpandProperty OwningProcess

# Check if any processes are found
if ($pids) {
    Write-Host "Found processes using port ${port}: ${pids}"

    # Kill each process using the port
    foreach ($pid in $pids) {
        try {
            Stop-Process -Id $pid -Force
            Write-Host "Successfully killed process with PID: $pid"
        } catch {
            Write-Host "Failed to kill process with PID: $pid. Error: $_"
        }
    }
} else {
    Write-Host "No processes found using port $port"
}


cd ./nomelmatungiss

# Create a virtual environment if it doesn't exist
if (-Not (Test-Path -Path "/.venv")) {
    python -m venv /.venv
}

# Activate the virtual environment
& ./.venv/Scripts/Activate.ps1
# move to the

# Pass the user locale to the Python script
$locale = $userLocale.Name

# Move to the directory containing your scripts
#cd ./nomelmatungiss
#install the requirements
pip install -r requirements.txt

# Start the server in a new background job
Start-Job -ScriptBlock { python .\example.py }

# Start the keyboard input listener script
python .\send_request.py $locale