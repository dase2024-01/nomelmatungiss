import requests
import pyperclip
from pynput.keyboard import  Controller, Key, KeyCode
# import keyboard
# keyboard.press(Key.space)
# keyboard.release(Key.space)


from pynput.keyboard import Key, Listener
file_to_open = 'config/config.json'
code_key_combination = 'kg'

current_keys = set()
COMBINATION = set()
for key in code_key_combination:
    COMBINATION.add(KeyCode(char=key))
def display_password(password):
    if password:
        pyperclip.copy(password)
        print("Password copied to clipboard: ", password)

def get_password():
    url = 'http://localhost:8000/generate-password'
    password_successfully_generated = False
    response = requests.get(url)
    while not password_successfully_generated:
        if response.status_code == 200:
            password = response.json().get('password')
            password_successfully_generated = True
            return password
        elif response.status_code == 500:
            print(' could not generate password ')
        elif response.status_code == 404:
            print('service is not erected')
            break
# COMBINATION = {KeyCode(char='k'), KeyCode(char='g')}
def on_press(key):
    print('key pressed', key)
    if key in COMBINATION:
        print('key pressed', key)
        current_keys.add(key)
        if all(k in current_keys for k in COMBINATION):
            print('key pressed', key)
            print('launching password generation')
            password = get_password()
            display_password(password)

def on_release(key):
    try:
        current_keys.remove(key)
    except KeyError:
        pass

# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()
def on_hotkey():
    password = get_password()
    display_password(password)




        # Here you can use additional methods to show the password at cursor location

if __name__ == '__main__':

    pass
    # keyboard.add_hotkey('shift+option+cmd+g', on_hotkey)
    # keyboard.wait('esc')
    # password = get_password()
    # display_password(password)