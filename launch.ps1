# Start the server in a new background job
Start-Job -ScriptBlock { python .\example.py }

# Start the keyboard input listener script
python .\send_request.py