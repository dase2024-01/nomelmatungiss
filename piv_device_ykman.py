# from ykman.device import list_all_devices
# from yubikit.core.smartcard import SmartCardConnection
# # identifying yubikeys
# for device, info in list_all_devices():
#     if info.version >= (5, 0, 0):  # The info object provides details about the YubiKey
#         print(f"Found YubiKey with serial number: {info.serial}")


from ykman.device import list_all_devices
from yubikit.core.smartcard import SmartCardConnection
from yubikit.piv import PivSession

# Select a connected YubiKeyDevice
dev, info = list_all_devices()[0]

# Connect to a YubiKey over a SmartCardConnection, which is needed for PIV.
with dev.open_connection(SmartCardConnection) as connection:
    # The connection will be automatically closed after this block

    piv = PivSession(connection)
    attempts = piv.get_pin_attempts()
    # print(f"You have 3 PIN attempts left.")
    print(f"You have {attempts.real} PIN attempts left.")

#     attempt to identify in the
from ykman.device import list_all_devices, scan_devices
from yubikit.core.smartcard import SmartCardConnection
from time import sleep

handled_serials = set()  # Keep track of YubiKeys we've already handled.
state = None

while True:  # Run this until we stop the script with Ctrl+C
    pids, new_state = scan_devices()
    if new_state != state:
        state = new_state  # State has changed
        for device, info in list_all_devices():
            if info.serial not in handled_serials:  # Unhandled YubiKey
                print(f"Programming YubiKey with serial: {info.serial}")
                ...  # Do something with the device here.
                handled_serials.add(info.serial)
    else:
        sleep(1.0)  # No change, sleep for 1 second.