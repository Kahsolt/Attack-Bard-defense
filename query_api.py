#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/10

# 迁移攻击：用本地攻击结果查询百度云API服务
# 百度智能云千帆大模型平台 API 介绍
# https://cloud.baidu.com/doc/WENXINWORKSHOP/s/flfmc9do2
# 多模态图像理解模型 Fuyu-8B
# https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Qlq4l7uw6
# https://cloud.baidu.com/doc/WENXINWORKSHOP/s/nlrobz49b

import os
import json
import yaml
import base64
from time import sleep
from pathlib import Path
from pprint import pprint as pp
from argparse import ArgumentParser
from traceback import print_exc

import requests as R
from utils.record_db import *

QUERY_PROMPTS = [
  'describe the picture',
  'describe the picture briefly',
  'describe the picture briefly, within 10 words',
  'describe the picture in details',
  'describe the picture in details as much as possible, more than 150 words',
]

PROVIDER = 'fuyu_8b'

API_EP_FREE = 'fuyu_8b'
API_URL = None
HEADERS = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

http = R.Session()


def setup(key_fp:Path, use_free:bool=False):
  global API_URL

  if key_fp.exists():
    with open(key_fp, 'r', encoding='utf-8') as fh:
      data = yaml.safe_load(fh)
      API_KEY = data['API_KEY']
      SECRET_KEY = data['SECRET_KEY']
      API_EP = data['API_EP']

  API_KEY = os.getenv('API_KEY', API_KEY)
  SECRET_KEY = os.getenv('SECRET_KEY', SECRET_KEY)
  assert API_KEY, '>> missing API_KEY'
  assert SECRET_KEY, '>> missing SECRET_KEY'
  print('>> API_KEY:', API_KEY)
  print('>> SECRET_KEY:', SECRET_KEY)

  TOKEN_URL = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_KEY}&client_secret={SECRET_KEY}'
  resp = http.post(TOKEN_URL, headers=HEADERS)
  data: Dict = resp.json()
  if 'error' in data:
    print('>> error:', data.get('error'))
    print('>> error_description:', data.get('error_description'))
    exit(-1)

  ACCESS_TOKEN = data.get('access_token')
  assert ACCESS_TOKEN, '>> missing ACCESS_TOKEN'
  print('>> ACCESS_TOKEN:', ACCESS_TOKEN)

  # fallback to the free API endpoint
  API_EP = os.getenv('API_EP', API_EP)
  API_EP = API_EP or API_EP_FREE
  if use_free: API_EP = API_EP_FREE
  assert API_EP, '>> missing API_EP'
  print('>> API_EP:', API_EP)

  API_URL = f'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/image2text/{API_EP}?access_token={ACCESS_TOKEN}'
  print('>> API_URL:', API_URL)
  print('=' * 72)


def query(fp:Path, prompt:str) -> Dict:
  payload = {
    'prompt': prompt,
    'image': base64.b64encode(read_file(fp)).decode(encoding='utf-8'),
  }
  try:
    resp = http.post(API_URL, headers=HEADERS, json=payload, timeout=10)
    data: Dict = resp.json()
  except KeyboardInterrupt:
    raise KeyboardInterrupt
  except:
    print_exc()
    return

  if 'error_code' in data:
    print('>> error_code:', data.get('error_code'))
    print('>> error_msg:', data.get('error_msg'))
    if data.get('error_code') == 17:  # Open api daily request limit reached
      exit(-1)
    sleep(1)
    return

  print('>> resp:')
  pp(data)
  return data


def walk(dp:Path):
  for p in dp.iterdir():
    if p.is_dir():
      yield from walk(p)
    else:
      yield p.resolve()


def run(args):
  fps = list(walk(args.img_dp))
  fps.sort()  # somewhat keep ordered
  if args.limit > 0: fps = fps[:args.limit]

  tot, ok, err, ign = 0, 0, 0, 0
  db = RecordDB(args.db_fp)
  try:
    prompt = QUERY_PROMPTS[args.pid]
    pid = db.get_prompt_id(prompt)
    for fp in tqdm(fps):
      tot += 1
      iid = db.get_image_id(fp)
      if args.ignore_queried and db.has(pid, iid):
        ign += 1
        continue

      ts_req = now()
      res = query(fp, prompt)
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
  parser.add_argument('-I', '--img_dp', default='outputs', type=Path, help='image folder, will walk recursively')
  parser.add_argument('-L', '--limit', default=-1, type=int, help='limit input image count')
  parser.add_argument('-O', '--db_fp', default=DB_FILE, type=Path, help='query record database file')
  parser.add_argument('-K', '--key_fp', default='API_KEY.yaml', type=Path, help='credential key file (*.yaml)')
  parser.add_argument('--ignore_queried', action='store_true', help='ignore query if already has records')
  parser.add_argument('--use_free', action='store_true', help='force use public free API endpoint')
  args = parser.parse_args()

  setup(args.key_fp, args.use_free)

  run(args)
