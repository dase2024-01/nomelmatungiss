import iso639

from PIL import ImageGrab
from collections import Counter
from fastapi import FastAPI, Request, Response
import re
app = FastAPI()
import subprocess
import os
import psutil
import uvicorn
import webcolors
from cryptography.fernet import Fernet
import sqlite3
import string

key = Fernet.generate_key()
cipher_suite = Fernet(key)
db_path = 'passwords.db'

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
from eng_syl.syllabify import Syllabel
x = 2
import transformers
import torch

ckpt_path = "internlm/internlm-xcomposer2-4khd-7b"

torch.set_grad_enabled(False)
# from langchain.llms import GPT4All
from gpt4all import GPT4All
# from iso639 import languages
import getpass
# iso639.

import json
CONFIG_FILE = 'config/config.json'
if CONFIG_FILE:
    with open(CONFIG_FILE) as f:
        password_config = json.load(f)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('example.log'), logging.StreamHandler()])
# iso639.to_iso639_2('lv')
full_language_name = iso639.to_name(password_config["language_code"])
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
    # Save the screenshot
    screenshot.save(save_path)
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
def get_recently_launched_apps(top_n=1):
    # Get the PID of the current script
    current_pid = os.getpid()

    # Get all running processes
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'create_time']):
        try:
            pinfo = proc.info
            pinfo['create_time'] = datetime.fromtimestamp(pinfo['create_time'])
            # Exclude the current script's process
            if pinfo['pid'] != current_pid:
                processes.append(pinfo)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    # Sort processes by creation time
    processes.sort(key=lambda p: p['create_time'], reverse=True)

    # Get the top N most recently launched processes
    recent_processes = processes[:top_n]

    return recent_processes


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
async def generate_password(request: Request):
    # get context
    example_info = get_recently_launched_apps(topN)

    latest_launcched_app = " ".join(example_info[0]['name'].split(' ')[:2])
    print(' latest launched app info ', latest_launcched_app)
    # get app

    example_rang = get_main_color()
    print(' colour of the screen ', example_rang)

    model_Llama3 = GPT4All(
                           model_name='Meta-Llama-3-8B-Instruct.Q4_0.gguf')
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



    print('output_Llama3', output_Llama3)
    check_pass = False
    generated_passphrases = re.findall(r'passphrase": "(.*?)"', output_Llama3)
    for passphrase in generated_passphrases:
        print(passphrase)
        split_words = re.findall('[A-Z][^A-Z]*', output_Llama3.split('passphrase": "')[1].split('"')[0])
        split_result = []
        for item in split_words:
            split_result.extend(item.split())
        # [x.split(' ') for x in split_words]
        no_words = len(split_result)
        candidate_phrase = " ".join(split_result)
        candidate_phrase_raw = candidate_phrase
        concat_phrase = candidate_phrase.replace(" ", "")
        no_chars = len(concat_phrase)
        # split('{')[1].split('}')
        if check_pass != False:
            break
        ent, strength = evaluate_password_strength(candidate_phrase)
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
                print('candidate_phrase', candidate_phrase)
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
    if password_config["replaceVowelsWithNumbers"]:
        for key in replacement_dict:
            candidate_phrase = candidate_phrase.replace(key, replacement_dict[key])
    # if there are diacritic marks, remove those from letters
    output_romanized = uroman.uroman(candidate_phrase)

    if password_config["capitalizeOutput"]:
        if password_config["capitalizeSyllables"]:
            syllabel = Syllabel()
            output_romanized = " ".join(syllabel.syllabify(output_romanized).split('-'))
        # else
        output_romanized = "".join([laten.capitalize() for laten in output_romanized.split(' ')])
    else:
        output_romanized = output_romanized.replace(' ', '')

    model_Llama3.close()

    print('Generated passphrase', output_romanized)

    response_payload = {"password": output_romanized,

                        "phrase": candidate_phrase_raw,
                        "strength": strength,
                        "entropy": ent,
                        }
    print(response_payload)
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







