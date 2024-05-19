#!/bin/bash

#change the directory

#cd /home/$(whoami)
#echo  /home/$(whoami)

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

#kill the process running on port 8000
lsof -ti:8000 | xargs kill -9
# Start the server in the background
python3 ./example.py &

# Start the keyboard input listener script
python3 ./send_request.py "$os_language"