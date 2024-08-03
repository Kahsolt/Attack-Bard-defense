from PIL import Image

import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor

device = 'cuda'

#pretrained_model_name_or_path = "llava-hf/llava-1.5-7b-hf"
pretrained_model_name_or_path = r'C:\Users\jiangsongru\.cache\huggingface\hub\models--llava-hf--llava-1.5-7b-hf\snapshots\9e7ca49a0174c7cc68c2605d221b4cac0aea799d'
model = LlavaForConditionalGeneration.from_pretrained(
  pretrained_model_name_or_path, 
  torch_dtype=torch.float16, 
  low_cpu_mem_usage=True, 
  #load_in_8bit=True,
  load_in_4bit=True,
  #use_flash_attention_2=True,
)#.to('cuda')
processor = LlavaProcessor.from_pretrained(
  pretrained_model_name_or_path, 
)

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
  {
    "role": "user",
    "content": [
      {"type": "text", "text": "What are these?"},
      {"type": "image"},
    ],
  },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)


try:
  while True:
    fp = input('>> input file path: ').strip()

    try:
      raw_image = Image.open(fp).convert('RGB')
    except Exception as e:
      print(e)
      continue

    inputs = processor(prompt, raw_image, return_tensors='pt').to(device, torch.float16)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    result = processor.decode(output[0][2:], skip_special_tokens=True)
    print(result)
except KeyboardInterrupt:
  pass
