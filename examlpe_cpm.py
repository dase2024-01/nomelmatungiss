import os

# from transformers.image_utils import load_image

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel, AutoModelForVision2Seq, AutoProcessor

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# b
model = AutoModel.from_pretrained( 'openbmb/MiniCPM-V-2',
                                   trust_remote_code=True,
                                   torch_dtype=torch.float16)
# For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
# model = model.to(device='cuda', dtype=torch.bfloat16)

# model = model.to(device='mps', dtype=torch.float16)


tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True)
model.eval()

image = Image.open('example.webp').convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': question}]


res, context, _ = model.chat(
    image=image,
    msgs=msgs,
    context=None,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7
)

print(res)