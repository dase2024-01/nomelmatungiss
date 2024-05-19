
# for Linux
#AUTOSTART_DIR="$HOME/.config/autostart"
#AUTOSTART_FILE="$AUTOSTART_DIR/launch_password_manager.desktop"
# for Mac

# Determine the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LAUNCH_SCRIPT_PATH="$SCRIPT_DIR/launch.sh"
DB_PATH="$SCRIPT_DIR/passwords.db"
PLIST_FILE="com.yourusername.launchpasswordmanager.plist"
PLIST_PATH="$HOME/Library/LaunchAgents/$PLIST_FILE"

#function to create on macos
create_plist() {
    cat <<EOF > "$PLIST_FILE"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.yourusername.launchpasswordmanager</string>
    <key>ProgramArguments</key>
    <array>
        <string>$LAUNCH_SCRIPT_PATH</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/launchpasswordmanager.out</string>
    <key>StandardErrorPath</key>
    <string>/tmp/launchpasswordmanager.err</string>
</dict>
</plist>
EOF
}

create_database() {
    if [ ! -f "$DB_PATH" ]; then
        sqlite3 "$DB_PATH" <<EOF
CREATE TABLE passwords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    password TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    app_name TEXT NOT NULL,
    app_hint TEXT DEFAULT NULL
);
EOF
        echo "Database created at $DB_PATH"
    else
        echo "Database already exists at $DB_PATH"
    fi
}



echo "This script will register the launch.sh script to run at startup."
read -p "Do you want to continue? (y/n) " consent

# install dependencies
cd $SCRIPT_DIR || return

pip3 install -r requirements.txt

if [[ "$consent" == "y" || "$consent" == "Y" ]]; then
    # Ensure the launch script is executable
    chmod +x "$LAUNCH_SCRIPT_PATH"

    # Check for macOS vs Linux
    OS_TYPE="$(uname)"
    if [[ "$OS_TYPE" == "Darwin" ]]; then
      # macOS specific setup
        create_plist

        # Move the plist file to the LaunchAgents directory
        mv "$PLIST_FILE" "$PLIST_PATH"


        # Load the Launch Agent using bootstrap
        launchctl bootout gui/$(id -u) "$PLIST_PATH" 2>/dev/null
        launchctl bootstrap gui/$(id -u) "$PLIST_PATH"
#        # Load the Launch Agent
#        launchctl load "$PLIST_PATH"
    elif [[ "$OS_TYPE" == "Linux" ]]; then
      # Linux specific setup
        AUTOSTART_DIR="$HOME/.config/autostart"
        AUTOSTART_FILE="$AUTOSTART_DIR/launch_password_manager.desktop"

        # Create the autostart directory if it doesn't exist
        mkdir -p "$AUTOSTART_DIR"

        # Create the autostart .desktop file
        cat <<EOF > "$AUTOSTART_FILE"
[Desktop Entry]
Type=Application
Exec=$LAUNCH_SCRIPT_PATH
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
Name=Launch Password Manager
Comment=Launch the password manager script at startup
EOF

        create_database
        echo "The launch script has been registered to run at startup on Linux."
    else
        echo "Unsupported OS: $OS_TYPE. Installation aborted."
    fi

#    # Load the .plist file
#    # Move the plist file to the LaunchAgents directory
#    mv "$PLIST_FILE" "$PLIST_PATH"

    # Load the Launch Agent
#    launchctl load "$PLIST_PATH"
    echo "The script has been registered to run at startup."
else
    echo "The script has not been registered to run at startup."
fi

#for Linux
    # Create the autostart .desktop file
#    cat <<EOF > "$AUTOSTART_FILE"
#[Desktop Entry]