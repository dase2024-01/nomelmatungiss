#!/bin/bash

# install dependencies
pip3 install -r requirements.txt

#kill the process running on port 8000
lsof -ti:8000 | xargs kill -9
# Start the server in the background
python3 ./example.py &

# Start the keyboard input listener script
python3 ./send_request.py