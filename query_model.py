#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/13

# 本地攻击结果查询本地开源模型

import os
import json
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
from traceback import print_exc

import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from utils.record_db import *

# NOTE: you may change to local folder path
pretrained_model_name_or_path = os.getenv('MODEL_NAME_OR_PATH', "llava-hf/llava-1.5-7b-hf")

PROVIDER = 'llava-1.5-7b-hf'
DB_FILE = LOG_PATH / 'record-llava.db'

QUERY_PROMPTS = [
  'describe the picture',
  'describe the picture briefly',
  'describe the picture briefly, within 10 words',
  'describe the picture in details',
  'describe the picture in details as much as possible, more than 150 words',
]


class ModelService:

  def __init__(self) -> None:
    self.model = LlavaForConditionalGeneration.from_pretrained(
      pretrained_model_name_or_path, 
      torch_dtype=torch.float16, 
      low_cpu_mem_usage=True, 
      #load_in_8bit=True,
      load_in_4bit=True,
      #use_flash_attention_2=True,
    )
    self.processor = LlavaProcessor.from_pretrained(
      pretrained_model_name_or_path, 
    )

  def query(self, fp:Path, prompt:str='What are these?') -> str:
    conversation = [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": prompt},
          {"type": "image"},
        ],
      },
    ]
    prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
    raw_image = Image.open(fp).convert('RGB')
    inputs = self.processor(prompt, raw_image, return_tensors='pt').to(device, torch.float16)
    output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
    result = self.processor.decode(output[0], skip_special_tokens=True)
    prybar = 'ASSISTANT:'
    return result[result.index(prybar) + len(prybar):].strip()


def walk(dp:Path):
  for p in dp.iterdir():
    if p.is_dir():
      yield from walk(p)
    else:
      yield p.resolve()


def run(args):
  if args.recursive:  # somewhat keep ordered, for deep recursive folder
    fps = list(walk(args.img_dp))
    fps.sort()
  else:               # keep ordered, for single flatten folder
    fps = list(Path(args.img_dp).iterdir())
    fps.sort(key=img_fps_sort_fn)
  if args.limit > 0: fps = fps[:args.limit]

  service = ModelService()

  tot, ok, err, ign = 0, 0, 0, 0
  db = RecordDB(args.db_fp)
  try:
    prompt = QUERY_PROMPTS[args.pid]
    pid = db.get_prompt_id(prompt)
    for fp in tqdm(fps):
      tot += 1
      iid = db.get_image_id(fp)
      if not args.allow_duplicate and db.has(pid, iid):
        ign += 1
        continue

      ts_req = now()
      res = service.query(fp, prompt)
      ts_res = now()
      if res is None:
        print('>> error query:', fp)
        err += 1
        continue

      db.add(pid, iid, json.dumps(res), ts_req, ts_res, PROVIDER)
      ok += 1
  except KeyboardInterrupt:
    print('>> Exit by Ctrl+C')
  except:
    print_exc()
  finally:
    db.close()

  print(f'>> [Done] tot={tot}, ok={ok}, err={err}, ign={ign}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-P', '--pid', default=0, type=int, help='query prompt id in the predefined list')
  parser.add_argument('-I', '--img_dp', default='outputs', type=Path, help='image folder')
  parser.add_argument('-L', '--limit', default=-1, type=int, help='limit input image count')
  parser.add_argument('-O', '--db_fp', default=DB_FILE, type=Path, help='query record database file')
  parser.add_argument('--recursive', action='store_true', help='walk recursively in --img_dp')
  parser.add_argument('--allow_duplicate', action='store_true', help='allow repeated query even if already has records')
  args = parser.parse_args()

  run(args)
