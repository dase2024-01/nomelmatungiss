import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

import json
import time
import random
import string
import getpass

username_for_payload = getpass.getuser()
# Example usage
user_id = username_for_payload
session_id = "session"

def generate_nonce(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
def generate_challenge_payload(user_id,
                               session_id):

    payload = {
        "nonce": generate_nonce(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_id": user_id,
        "session_id": session_id
    }
    return json.dumps(payload)

challenge_payload = generate_challenge_payload(user_id, session_id)

challenge_file = 'challenge.json'
with open(challenge_file, 'w') as f:
    f.write(challenge_payload)