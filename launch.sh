#!/bin/bash

#change the directory

#cd /home/$(whoami)
#echo  /home/$(whoami)
# Example command to demonstrate logging
echo "Running launch.sh at $(date)" >> /tmp/launchpasswordmanager.log
# Get the system language
# Get the system language based on the operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS specific command to get the system language
  os_language=$(defaults read -g AppleLocale | cut -d '_' -f1)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  # Linux specific command to get the system language
  if command -v localectl &> /dev/null; then
    os_language=$(localectl status | grep "System Locale" | cut -d '=' -f2 | cut -d '_' -f1)
  else
    os_language=$(locale | grep LANG= | cut -d= -f2 | cut -d_ -f1)
  fi
else
  echo "Unsupported OS type: $OSTYPE"
  exit 1
fi

# If the os_language is empty, set a default value
if [ -z "$os_language" ]; then
  os_language="en"
fi


cd ~/PycharmProjects/pythonProject
#kill the process running on port 8000
lsof -ti:8000 | xargs kill -9


#activate the virtual environment
source ./.venv/bin/activate/
#.venv/bin/python3 -m pip install -r requirements.txt

# Start the server in the background
.venv/bin/python3 ./example.py & >> ~/Desktop/server_log.txt

# Start the keyboard input listener script
.venv/bin/python3 ./send_request.py "$os_language" >> ~/Desktop/keyboard_log.txt