# Secure Memorable Password Generator

This project is implemented as the inference implementation of a generator 
of secure and memorable passwords on the Business Informatics masters study program
during the year 2023-24. 

## Minimal system requirements:

- Python 3.9 (or higher, lower versions may work but are not tested)
- 8GB of RAM (for decent performance, 16 GB or more is recommended)
- At least 8 GB of free disk space
- At least Mac Big Sur (for MacOS) or Windows. Linux: Not fully supported.

## Required permissions: 

These need to be given to the application in which the launch scripts are run: 
- Access to the internet (for downloading the required libraries and for resolving the LLM models)
- Access to the file system (for saving the generated passwords)
- Access to the clipboard (for copying the generated passwords)
- Access to screen capturing (for analyzing the visual content)
- Access to reading the keyboard input (for analyzing the keyboard input)

- (Do not worry, the application will ask for these permissions during installation,
and no data will be collected or sent to any third parties)

## Installation:

1. `$ git clone https://github.com/dase2024-01/nomelmatungiss`
2. `chmod +x install.sh`
2. `$ install.sh`

## Launch:

1. `$ launch.sh`
2. At the first run, the app needs to download the required libraries and weights for the LLM models. 
Dependent on the internet connection , it can take up to several hours to complete.

## Usage: 

When correctly installed, the app should run as background process reading the keystrokes with an LLM server on default port 8000.

It has the keystroke 'k+g' (default) as a trigger to forcibly generate the password for the current window (of application), 
and 'k+j' to retrieve the  hint for a password (if exists) or to generate a new password (if not). These keys can be changed in config.json.

Passwords are securely stored in the sqlite3 database, the key being stored in the PC keyring. 

The hint for a password is a base "meaningful" sentence before its romanization and replacement per the transformation lexicon specified in the configs.
After setting their own transformation lexicon, the user is supposed to perceive the seed passphrase through its masked rendition, 
and the hint is supposed to help in the recall process of a password . As a consequence, the user rids of need to lookup the password,
and the password is never written in the wrong place.

It takes the locale language of the system as basis for password generation. Most alphabetical official languages are supported, 
but convenience decreases for minor and abjadic languages as opposed to widespread and fully vocalized ones. 



## Limitations:

1. The application does not dig into browser tabs and does not take their names into account. 
2. No GUI interface.
3. Passwords are generated per device, no syncing is available. 


## Behaviour Configuration:

This configuration file, .json, is for a password generator application.
Below are the details of the configuration parameters and their functions:

the app name for which the password is generated. By default left empty, 
then the latest app launched is perceived as the application. 
`app_name: "Word"`

The name of the application generating the passwords.

`language_code: "lv"`

The language code, set to Latvian (lv).

`context: "BLUE"`

The context in which the password generator will be used. This can be used to set specific settings or themes.

`MIN_WORDS: 4`

The minimum number of words to be included in the generated password.

`MAX_SYMBOLS: 25`

The maximum number of symbols allowed in the generated password.

`MIN_SYMBOLS: 13`

The minimum number of symbols required in the generated password.

`capitalizeOutput: false`

A boolean flag indicating whether the output password should be fully capitalized. Set to false, meaning no capitalization will be applied to the entire password.

`capitalizeSyllables: false`

A boolean flag indicating whether each syllable in the generated password should be capitalized. Set to false, meaning syllables will not be capitalized.

`replaceVowelsWithNumbers: true`

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

`replaceGeminatedLetters: true`

A boolean flag indicating whether geminated (doubled) letters should be replaced. Set to true, meaning geminated letters will be replaced.
itemToReplaceGeminatedLetters: "_"

The character to replace geminated letters with. In this configuration, geminated letters will be replaced with an underscore (_).


## More granular installation:
2. `venv/bin/activate`

3. `pip install -r requirements.txt`

4. `$ python3 main.py`

## License 
MIT.