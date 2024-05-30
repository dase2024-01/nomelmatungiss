import os

# from transformers.image_utils import load_image

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel, AutoModelForVision2Seq, AutoProcessor

import torch
torch.cuda.empty_cache()
from PIL import Image
from transformers import AutoModel, AutoTokenizer
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# b
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S', handlers=[logging.FileHandler('example.log'), logging.StreamHandler()])
if torch.cuda.is_available():
    print('CUDA available')

    device_name = 'cuda'
else:
    device_name = 'mps'
model = AutoModel.from_pretrained( 'openbmb/MiniCPM-V-2',
                                   trust_remote_code=True,
                                   torch_dtype=torch.float16)
# For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
model = model.to(device=device_name, dtype=torch.float16)

# model = model.to(device='mps', dtype=torch.float16)


tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True)
model.eval()

# image = Image.open('example.webp').convert('RGB')
image = Image.open('2024-05-19 19.06.52.jpg').convert('RGB')
question = 'What color is the picture? Answer in English.'
question2 = 'What can you see in the picture? Answer in English'
msgs = [{'role': 'user', 'content': question2}]


res, context, _ = model.chat(
    image=image,
    msgs=msgs,
    context=None,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.2
)
logging.info(f"""{res}, {context}, is the context of the picture """)
# print(res)