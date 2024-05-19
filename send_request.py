import time
import sys
import requests
import pyperclip
from pynput.keyboard import  Controller, Key, KeyCode
# import keyboard
# keyboard.press(Key.space)
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('example.log'),
                              logging.StreamHandler()])
# keyboard.release(Key.space)
import locale
import iso639
import json
CONFIG_FILE = 'config/config.json'
if CONFIG_FILE:
    with open(CONFIG_FILE) as f:
        password_config = json.load(f)
if 'dbPath' not in password_config:
    db_path = password_config["dbPath"]
else:
    db_path = 'passwords.db'


from pynput.keyboard import  Listener
file_to_open = 'config/config.json'
code_key_combination_for_generation = 'kg'
code_key_combination_for_hint = 'kj'

current_keys = set()
COMBINATION_GEN = set()
for key in code_key_combination_for_generation:
    COMBINATION_GEN.add(KeyCode(char=key))

COMBINATION_HINT = set()
for key in code_key_combination_for_hint:
    COMBINATION_HINT.add(KeyCode(char=key))
def display_password(password):
    if password:
        pyperclip.copy(password)
        print("Password copied to clipboard: ", password)

def get_password(regenerate=True):
    import logging
    logging.info(sys.argv)
    language_of_locale = str(sys.argv[1])
    # logging.info(' type of the argument {}'.format(type(language_of_locale)))
    # [0]
    logging.info(f' language of the locale {language_of_locale}, language code {password_config["language_code"]} ', )
    # iso639.to_iso639_2('lv')
    if ('language_code' not in password_config) or (password_config["language_code"] == ''):
        full_language_name = iso639.to_name(language_of_locale)
    else:
        full_language_name = iso639.to_name(password_config["language_code"])
    full_language_name= full_language_name.split('; ')[0]
    url = 'http://localhost:8000/generate-password?regenerate={}&full_language_name={}'.format(regenerate, full_language_name)
    password_successfully_generated = False


    response = requests.get(url)
    # locale.setlocale(locale.LC_ALL, '')
    # get_language_of_locale = locale.getdefaultlocale()[0].split('_')[0]

    while not password_successfully_generated:
        if response.status_code == 200:
            if regenerate == False:
                print('hint password generated')
                password = response.json().get('hint')
            else:
                print('password generated')
                password = response.json().get('password')
            password_successfully_generated = True
            return password
        elif response.status_code == 500:
            print(' could not generate password ')
            time.sleep(4)
        elif response.status_code == 404:
            print('service is not erected')
            break
# COMBINATION = {KeyCode(char='k'), KeyCode(char='g')}
def on_press(key):
    print('key pressed', key)
    if key in COMBINATION_GEN:
        print('key pressed', key)
        current_keys.add(key)
        if all(k in current_keys for k in COMBINATION_GEN):
            print('key pressed', key)
            print('launching password generation')
            password = get_password(regenerate=True)
            password = password.encode('latin-1').decode('utf-8')
            display_password(password)


    if key in COMBINATION_HINT:
        print('key pressed', key)
        current_keys.add(key)
        if all(k in current_keys for k in COMBINATION_HINT):
            print('key pressed', key)
            print('launching password hint retrieval')
            password = get_password(regenerate=False)
            try:
                password = password.encode('latin-1').decode('utf-8')
            except Exception as e:
                password = password
            display_password(password)
            # Here you can use additional methods to show the password at cursor location

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