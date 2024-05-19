# Secure Memorable Password Generator

This project is implemented as the inference implementation of a generator 
of secure and memorable passwords on the Business Informatics masters study program
during the year 2023-24. 

## Minimal system requirements:

- Python 3.9 (or higher, lower versions may work but are not tested)
- 8GB of RAM (for decent performance, 16 GB or more is recommended)
- At least 8 GB of free disk space


## Behaviour Configuration:

This configuration file, .json, is for a password generator application named "Word". Below are the details of the configuration parameters and their functions:

app_name: "Word"

The name of the application generating the passwords.
language_code: "lv"

The language code, set to Latvian (lv).
context: "BLUE"

The context in which the password generator will be used. This can be used to set specific settings or themes.
MIN_WORDS: 4

The minimum number of words to be included in the generated password.
MAX_SYMBOLS: 25

The maximum number of symbols allowed in the generated password.
MIN_SYMBOLS: 13

The minimum number of symbols required in the generated password.
capitalizeOutput: false

A boolean flag indicating whether the output password should be fully capitalized. Set to false, meaning no capitalization will be applied to the entire password.
capitalizeSyllables: false

A boolean flag indicating whether each syllable in the generated password should be capitalized. Set to false, meaning syllables will not be capitalized.
replaceVowelsWithNumbers: true

A boolean flag indicating whether vowels should be replaced with numbers according to the vowelReplacementDictionary. Set to true, meaning vowels will be replaced.
vowelReplacementDictionary:

A dictionary mapping specific vowels to their replacement numbers or symbols:
"a": "4"
"g": "9"
"e": "5"
"ī": "1"
"o": "0"
"u": "#"
"ž": "3"
"š": "$"
"y": "*"
"q": "9"
"i": "!"
"ē": "&"
"ā": "@"
replaceGeminatedLetters: true

A boolean flag indicating whether geminated (doubled) letters should be replaced. Set to true, meaning geminated letters will be replaced.
itemToReplaceGeminatedLetters: "_"

The character to replace geminated letters with. In this configuration, geminated letters will be replaced with an underscore (_).

## Installation:

Default server port is 8000 

1. `$ git clone https://github.com/dase2024-01/nomelmatungiss`
2. `$ launch.sh`

## More granular installation:
2. `venv/bin/activate`

3. `pip install -r requirements.txt`

4. `$ python3 main.py`