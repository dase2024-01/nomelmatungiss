import sys

import iso639

import sys
if sys.platform == 'win32' or sys.platform.startswith('win'):
    # import win32
    pass
else:
    from cryptography.fernet import Fernet
    import objc
    import Cocoa


# Cocoa.NSB
from PIL import ImageGrab
from collections import Counter
from fastapi import FastAPI, Request, Response
import re
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import locale

# example_info = get_recently_launched_apps(topN)

import keyring
import os
KEY_FILE_PATH = 'secret.key'
SERVICE_NAME = 'PasswordManager'
KEY_NAME = 'encryption_key'
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('example.log'),
                              logging.StreamHandler()])

import platform

# def get_current_input_source_mac():
#     import Quartz
#     import Foundation
#     source = Quartz.TISCopyCurrentKeyboardInputSource()
#     source_id = Quartz.TISGetInputSourceProperty(source, Quartz.kTISPropertyInputSourceID)
#     return source_id
#
# def get_current_input_source_windows():
#     import ctypes
#     import locale
#     user32 = ctypes.WinDLL('user32', use_last_error=True)
#     hkl = user32.GetKeyboardLayout(user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None))
#     lid = hkl & 0xFFFF
#     return locale.windows_locale.get(lid, f'Unknown (0x{lid:X})')

# def get_current_input_source():
#     if platform.system() not in ['Darwin', 'Windows']:
#         raise NotImplementedError("This function is only implemented for macOS and Windows")
#     elif platform.system() == 'Darwin':
#         return get_current_input_source_mac()
#     elif platform.system() == 'Windows':
#         return get_current_input_source_windows()

def get_active_window_mac():
    import Quartz
    window_info = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly,
                                                    Quartz.kCGNullWindowID)
    for window in window_info:
        if window.get('kCGWindowLayer') == 0:  # Check if it's the top-level window
            return window
    return None

def get_active_window_windows():
    import pygetwindow as gw
    import win32process
    import win32gui
    hwnd = win32gui.GetForegroundWindow()
    return hwnd

def get_process_for_active_window_windows():
    import win32process
    hwnd = get_active_window_windows()
    if not hwnd:
        return None

    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    try:
        process = psutil.Process(pid)
        app_name = process.name()
        return process , app_name
    except psutil.NoSuchProcess:
        return None
def get_active_window():
    if platform.system() not in ['Darwin', 'Windows']:
        raise NotImplementedError("This function is only implemented for macOS and Windows")
    elif platform.system() == 'Darwin':
        return get_active_window_mac()


def contains_latin_characters(password):
    # Regular expression pattern to match Latin characters (A-Z, a-z)
    pattern = re.compile(r'[A-Za-z]')

    # Search for the pattern in the password
    if pattern.search(password):
        return True
    else:
        return False

def get_process_for_active_window_mac():
    active_window = get_active_window_mac()
    if not active_window:
        return None

    pid = active_window['kCGWindowOwnerPID']
    app_name = active_window['kCGWindowOwnerName']
    try:
        process = psutil.Process(pid)
        return process, app_name
    except psutil.NoSuchProcess:
        return None, app_name

def get_process_for_active_window():
    if platform.system() == 'Darwin':
        return get_process_for_active_window_mac()
    elif platform.system() == 'Windows':
        return get_process_for_active_window_windows()
if sys.platform != 'win32':
    pass
    def generate_and_store_key():
        """
        Generate a new key and save it to a file.
        """
        import Fernet
        key = Fernet.generate_key()
        keyring.set_password(SERVICE_NAME, KEY_NAME, key.decode())
        logging.info(f'key generated {key}')
        return key

    # uncomment to save the key to a file
    # with open(KEY_FILE_PATH, 'wb') as key_file:
            #     key_file.write(key)
    def load_key():
        """
        Load the key from the key file.
        """
        key = keyring.get_password(SERVICE_NAME, KEY_NAME)
        if key is None:
            # If the key does not exist, generate and store a new one
            key = generate_and_store_key()
        else:
            key = key.encode()
        logging.info(f'key retrieved {key}')
        return key

    # uncomment to save the key to a file
    # return open(KEY_FILE_PATH, 'rb').read()

    if not os.path.exists(KEY_FILE_PATH):
        generate_and_store_key()

import subprocess
import os
import psutil
import uvicorn
import webcolors

import sqlite3

app = FastAPI()
if sys.platform != 'win32':
    key = load_key()
    cipher_suite = Fernet(key)
else:
    pass
# key = Fernet.generate_key()



# processes = [p.info for p in psutil.process_iter(['pid', 'name', 'create_time'])]
# # Sort processes by creation time
# processes.sort(key=lambda p: p['create_time'], reverse=True)
# current_pid = os.getpid()
# # Get the top N most recently launched processes
# recent_processes = processes[:topN]
#
# # Get the most recently launched process
# recent_process = processes[0]


# print(f"Most recently launched application: {recent_process['name']}")
if sys.platform != 'win32':
    from eng_syl.syllabify import Syllabel
x = 2
import transformers
import torch



torch.set_grad_enabled(False)
# from langchain.llms import GPT4All
from gpt4all import GPT4All
# from iso639 import languages

import getpass
# iso639.

import json
ckpt_path = "internlm/internlm-xcomposer2-4khd-7b"
CONFIG_FILE = 'config/config.json'
if CONFIG_FILE:
    with open(CONFIG_FILE) as f:
        password_config = json.load(f)
if 'dbPath' not in password_config:
    db_path = password_config["dbPath"]
else:
    db_path = 'passwords.db'


# name_ iso639.to_name('lv')
import os
import uroman
import math
user_name_of_pc = getpass.getuser()
name_of_model_lmstudio = 'Meta-Llama-3-8B-Instruct-GGUF'
name_of_model_gpt4all = 'Meta-Llama-3-8B-Instruct.Q4_0.gguf'
# LMStudio_filename = '/Users/garamantys/.cache/lm-studio/models/lmstudio-community'
GGML_REPO_DIR = f'/Users/{user_name_of_pc}/Library/Application Support/nomic.ai/GPT4All'
# GPT4All_filename = os.path.join
# uroman.uroman('āčēģīķļņšūž')
llm_model_filename_full = os.path.join(GGML_REPO_DIR, name_of_model_gpt4all)
# llm_model_filename_full = os.path.join(LMStudio_filename, name_of_model)
# GPT4All(model)
from datetime import datetime
import numpy as np

key = 'OM7z6ZqJ88kcqKu28nyz/4j+EeEM/2Ovpw8ESjqc6rg='

def get_ram_info():
    # Get the virtual memory statistics
    virtual_memory = psutil.virtual_memory()
    vram_in_gigabytes = virtual_memory.total / (1024 ** 3)
    return vram_in_gigabytes

def init_db():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    init_sql_script = '''
        CREATE TABLE IF NOT EXISTS passwords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            password TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
            app_name TEXT NOT NULL,
            app_hint TEXT DEFAULT NULL
        )
    '''
    c.execute(init_sql_script)
    conn.commit()
    conn.close()

def store_password(password, app_name, hint):
    if sys.platform.startswith('Windows'):
        return store_password_windows(password=password,
                                      key = key,
                                      app_name = app_name,
                                      hint = app_hint)
    else:
    #     Darwin
        return store_password_mac(encrypted_password = password,
                                  app_name=app_name,
                                      hint=app_hint)
    # return password

def encrypt_password_windows(password,
                           key,
                           app_name,
                           app_hint):
    # cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(password.encode('utf-8'))
    return base64.b64encode(nonce + tag + ciphertext).decode('utf-8')

def store_password(encrypted_password,
                   app_name,
                   app_hint=None):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    init_sql_script = '''
        CREATE TABLE IF NOT EXISTS passwords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            password TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
            app_name TEXT NOT NULL,
            app_hint TEXT DEFAULT NULL
        )
    '''
    c.execute(init_sql_script)


    c.execute('INSERT INTO passwords (password, app_name, app_hint) VALUES (?, ?, ?)', (encrypted_password,
                                                                                        app_name,
                                                                                        app_hint))
    conn.commit()
    conn.close()

def retrieve_passwords(db_path,
                       app_name,
                       app_hint):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        f"""SELECT  password, app_hint FROM passwords where app_name = '{app_name}' ORDER BY timestamp desc  LIMIT 1""")
    row = c.fetchall()
    conn.close()
    if row:
        encrypted_password = row[0][0]
    if sys.platform.startswith('Windows'):
        return retrieve_passwords_windows(encrypted_password = encrypted_password,
                                          key = key,
                                          app_name = app_name,
                                          hint = app_hint)
    else:
    #     Darwin
        return retrieve_passwords_mac(app_name=app_name,
                                      hint=app_hint)

def retrieve_passwords_windows( encrypted_password,
        key,
                                app_name,
                                hint = True):


        encrypted_data = base64.b64decode(encrypted_password)
        nonce = encrypted_data[:16]
        tag = encrypted_data[16:32]
        ciphertext = encrypted_data[32:]
        cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
        # return
        return ( cipher.decrypt_and_verify(ciphertext, tag).decode('utf-8'),
                     row[0][1])

def retrieve_passwords_mac(password ,
                           app_name,
                       hint = True):

    decrypted_password = cipher_suite.decrypt(token= password).decode()
    # decrypted_passwords.append((row[0], decrypted_password, row[2]))
    return ( decrypted_password,
             row[0][1])
    # else:
    #     return None

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

# main colour
def get_main_color():
    # Capture the entire screen
    screenshot = ImageGrab.grab()
    save_path = 'screenshot.png'
    # Save the screenshot (optional)kg
    # screenshot.save(save_path)
    # Convert the screenshot to a numpy array
    img_np = np.array(screenshot)

    # Reshape the array to be a list of RGB tuples
    pixels = img_np.reshape(-1, img_np.shape[-1])

    # Count the frequency of each color
    pixel_counts = Counter(map(tuple, pixels))

    # Get the most common color
    main_color = pixel_counts.most_common(1)[0][0]

    main_color_word = closest_color(main_color)

    return main_color_word



# def main_get_colour():
#     main_color = get_main_color()
#     print(f"The main color of the screen is: {main_color}")
# Number of recent applications to retrieve
topN = 1
# def get_recently_launched_apps(top_n=1):
#     # Get the PID of the current script
#     current_pid = os.getpid()
#     pattern = re.compile(r'^[A-Z]')
#     # Get all running processes
#     processes = []
#     for proc in psutil.process_iter(['pid', 'name', 'create_time']):
#         try:
#             pinfo = proc.info
#             pinfo['create_time'] = datetime.fromtimestamp(pinfo['create_time'])
#             # Exclude the current script's process
#             if pinfo['pid'] != current_pid:
#                 # processes.append(pinfo)
#                 if pattern.match(proc.info['name']):
#                     processes.append(proc.info)
#         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
#             pass
#     # processe
#     # Sort processes by creation time
#     processes.sort(key=lambda p: p['create_time'], reverse=True)
#
#     # Get the top N most recently launched processes
#     recent_processes = processes[:top_n]
#
#     return recent_processes


def calculate_password_entropy(password):
    # Determine the character set size
    charset_size = 0
    if any(c.islower() for c in password):
        charset_size += 26  # a-z
    if any(c.isupper() for c in password):
        charset_size += 26  # A-Z
    if any(c.isdigit() for c in password):
        charset_size += 10  # 0-9
    if any(c in '!@#$%^&*()-_=+[]{}|;:,.<>?/`~' for c in password):
        charset_size += 32  # Special characters (common ones)

    # Calculate the entropy
    entropy = len(password) * math.log2(charset_size)
    return entropy

def find_geminated_letters(text):
    # Regular expression to find geminated letters
    pattern = r'([a-zA-Z0-9])\1'
    return re.findall(pattern, text)
def evaluate_password_strength(password):
    entropy = calculate_password_entropy(password)
    # Define entropy thresholds for password strength categories
    if entropy < 28:
        strength = "Very Weak"
    elif entropy < 36:
        strength = "Weak"
    elif entropy < 60:
        strength = "Moderate"
    elif entropy < 128:
        strength = "Strong"
    else:
        strength = "Very Strong"
    return entropy, strength


# from gpt4all import AutoModel, AutoTokenizer
# model_name='Llama3',
import logging
# model_name='Llama3',
# model_path=llm_model_filename_full,
# currently_available_models = GPT4All.list_models()

@app.get("/generate-password")
async def generate_password(request: Request,
                            regenerate: bool = True,
                            full_language_name: str = 'German'):

    # get context
    if sys.platform != 'win32':
        # latest_launcched_app = " ".join(example_info[0]['name'].split(' ')[:2])
        process, latest_launcched_app = get_process_for_active_window()
        # current_input_method = get_current_input_source()
        # print(' current input method ', current_input_method)
        print(' latest launched app info ', latest_launcched_app)
    else:
        pass
    # get app

    # try to get password hint

    if regenerate == False:
        hint = retrieve_passwords(latest_launcched_app)
        if hint:
            print(f' found the passwords {hint}')
            response_payload = {"password": hint[0],
                                "hint": hint[1],
                                "strength": "Very Strong",
                                "entropy": 128,
                                }
            return Response(
                content=json.dumps(response_payload),
                            media_type="application/json"
            )

    example_rang = get_main_color()
    print(' colour of the screen ', example_rang)
    current_vram_storage = get_ram_info()
    if current_vram_storage > 10:
        fully_qualified_model_name = 'Meta-Llama-3-8B-Instruct.Q4_0.gguf'
    elif current_vram_storage > 8:
        # GGML_REPO_DIR = f'/Users/romulaperov/Library/Application Support/nomic.ai/GPT4All'
        fully_qualified_model_name = 'gpt4all-falcon-q4_0.gguf'


    model_Llama3 = GPT4All(model_name=fully_qualified_model_name)
    print(f' device used by the model {model_Llama3.device}')



    # Llama 3 Instruct
    app_name = 'Word'
    app_name = latest_launcched_app
    context_words = [example_rang]
    MIN_WORDS = 3
    MAX_SYMBOLS = 25
    MIN_SYMBOLS = 13
    # "{app_name}".
    #  and context: {" ".join(context_words)},
    user_prompt = f"""
    Create one memorable passphrase in {full_language_name} for the following application {app_name}
    and context: {" ".join(context_words)}. 
    Should be a meaningful sentence: Subject - verb - object
    at least {MIN_WORDS} words, at least {MIN_SYMBOLS} symbols, no more than {MAX_SYMBOLS} symbols 
    Write as {{"passphrase" : "your passphrase"}}. Return only passphrase. 
    """
    logging.info(f"Prompt: {user_prompt}")
    output_Llama3 = model_Llama3.generate(prompt = user_prompt,
                                          max_tokens=50, temp=0.7,
                                          repeat_penalty=1.18,
                                          top_k = 40, top_p = 0.4)



    # print('output_Llama3', output_Llama3)
    check_pass = False
    generated_passphrases = re.findall(r'passphrase": "(.*?)"', output_Llama3)
    # print('output_Llama3', generated_passphrases)
    for passphrase in generated_passphrases:
        print(f' the phrase is {passphrase} exanmple' )
        split_words = re.findall('[A-Z][^A-Z]*', output_Llama3.split('passphrase": "')[1].split('"')[0])
        print(f' the phrase is {passphrase} fff')
        # print('output_Llama3', split_words)
        split_result = []
        print(f' the phrase is {passphrase} fff')
        for item in split_words:
            split_result.extend(item.split())
        print(f' the phrase is {passphrase} fff')
            # print('output_Llama3', split_result)
        # [x.split(' ') for x in split_words]kg
        no_words = len(split_result)
        print('output_Llama3', split_result)
        print(f' the phrase is {passphrase} fff')
        if split_result != []:
            candidate_phrase = " ".join(split_result)
        else:
            print(f' the phrase is {passphrase} fff')
            candidate_phrase = passphrase
        # if candidate_phrase == '':
        #     candidate_phrase = passphrase
        print(f'output_Llama3{candidate_phrase} rrf ', )
        candidate_phrase_raw = candidate_phrase
        # print('output_Llama3', candidate_phrase)
        concat_phrase = candidate_phrase.replace(" ", "")
        no_chars = len(concat_phrase)
        # split('{')[1].split('}')
        if check_pass != False:
            break
        romanized = uroman.uroman(candidate_phrase)
        print(' analyzed phrase ', romanized)
        ent, strength = evaluate_password_strength(romanized)
        print('password strength ')
        while not check_pass:
            if no_words < MIN_WORDS:
                corr_prompt = f" should be at least {MIN_WORDS} words"
                check_pass = False
            elif no_chars > MAX_SYMBOLS:
                corr_prompt = f" should be no more than {MAX_SYMBOLS} symbols"
            elif strength not in ['Strong', 'Very Strong']:
                check_pass = False
                corr_prompt = f"make another passphrase per requirements, but with more symbols"
            else:
                check_pass = True

            if check_pass == False:
                print('candidate_phrase', uroman.uroman(candidate_phrase))
                print('could not pass the requirements')
                break
                # output_Llama3 = model_Llama3.generate(prompt=corr_prompt,
                #                                       max_tokens=50)
                # print(output_Llama3)

    replacement_dict = password_config["vowelReplacementDictionary"]
    if password_config["replaceGeminatedLetters"]:
        geminated_letters = find_geminated_letters(candidate_phrase)
        for letter in geminated_letters:
            candidate_phrase = candidate_phrase.replace(letter*2, letter + password_config["itemToReplaceGeminatedLetters"])
    print(' analyzed phrase ', candidate_phrase)

    if not contains_latin_characters(candidate_phrase):
        candidate_phrase = uroman.uroman(candidate_phrase)

    if password_config["replaceVowelsWithNumbers"]:
        for key in replacement_dict:
            candidate_phrase = candidate_phrase.replace(key, replacement_dict[key])
    print(' analyzed phrase ', candidate_phrase)
    # if there are diacritic marks, remove those from letters
    if contains_latin_characters(candidate_phrase):
        output_romanized = uroman.uroman(candidate_phrase)
    else:
        output_romanized = candidate_phrase
    # output_romanized = uroman.uroman(candidate_phrase)

    if password_config["capitalizeOutput"]:
        if (password_config["capitalizeSyllables"]) and (sys.platform == "win32"):
            syllabel = Syllabel()
            output_romanized = " ".join(syllabel.syllabify(output_romanized).split('-'))
        # else
        output_romanized = "".join([laten.capitalize() for laten in output_romanized.split(' ')])
    else:
        output_romanized = output_romanized.replace(' ', '')

    model_Llama3.close()
    # # Step 1: Re-encode the string to bytes using 'latin-1'
    # byte_sequence = candidate_phrase_raw.encode('latin-1')
    #
    # # Step 2: Decode the bytes back to a string using 'utf-8'
    # candidate_phrase_raw = byte_sequence.decode('utf-8')
    # candidate_phrase_raw = candidate_phrase_raw.encode(encoding='utf-8')
    print('Generated passphrase', output_romanized)
    print('Hint for passphrase', candidate_phrase_raw)

    response_payload = {"password": output_romanized,
                        "phrase": candidate_phrase_raw,
                        "strength": strength,
                        "entropy": ent,
                        }

    if 'win' in sys.platform:
        encrypted_password = encrypt_password_windows(password = output_romanized.encode('utf-8'),
                                                      key=key,
                                                      app_name=app_name)
    else:
        encrypted_password = cipher_suite.encrypt(output_romanized.encode())

    # dump password to db
    store_password(encrypted_password,
                   app_name,
                   app_hint = candidate_phrase_raw
                   )

    # print(response_payload)
    return Response(content=json.dumps(response_payload),
                    media_type="application/json")

if __name__ == '__main__':

    logging.info('Starting the server: http://localhost:8000/generate-password')
    uvicorn.run(app, host='localhost', port=8000)
#
# model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-4khd-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
# tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-4khd-7b', trust_remote_code=True)
# example_path = './example.webp'
# # ~/Downloads
# query1 = '<ImageHere>Illustrate the fine details present in the image'
# # image = './example.webp'
# # with torch.cuda.amp.autocast():
# response, _ = model.chat(tokenizer,
#                          query=query1,
#                          image=example_path,
#                          hd_num=55,
#                          history=[],
#                          do_sample=False,
#                          num_beams=3)
# print(response)
# # tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True).cuda()
# #
# # model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
#
#
# model_id = "meta-llama/Meta-Llama-3-8B"
#
# pipeline = transformers.pipeline("text-generation",
#                                  model=model_id,
#                                  model_kwargs={
#                                      "torch_dtype": torch.bfloat16
#                                  },
#                                  device_map="auto")
#
#
# pipeline("Hey how are you doing today?")
# # idefics xx2 8b
#
#







