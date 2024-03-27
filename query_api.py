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
import yaml
from pathlib import Path
from pprint import pprint as pp
from argparse import ArgumentParser
from traceback import print_exc

import requests as R
from utils.record_db import *

QUERY_PROMPT = '''
introduce the picture
'''
QUERY_PROMPT_CN = '''
分析一下图片画了什么
'''

API_EP = None
API_KEY = None
SECRET_KEY = None
ACCESS_TOKEN = None

HEADERS = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

http = R.Session()


def setup(key_fp:Path):
  global API_KEY, SECRET_KEY, ACCESS_TOKEN

  if key_fp.exists():
    with open(key_fp, 'r', encoding='utf-8') as fh:
      data = yaml.safe_load(fh)
      API_KEY = data['API_KEY']
      SECRET_KEY = data['SECRET_KEY']
      API_EP = data['API_EP']

  API_KEY = os.getenv('API_KEY', API_KEY)
  SECRET_KEY = os.getenv('SECRET_KEY', SECRET_KEY)
  API_EP = os.getenv('API_KEY', API_EP)

  assert API_KEY, '>> missing API_KEY'
  assert SECRET_KEY, '>> missing SECRET_KEY'
  assert API_EP, '>> missing API_EP'
  print('>> API_KEY:', API_KEY)
  print('>> SECRET_KEY:', SECRET_KEY)
  print('>> API_EP:', API_EP)

  url = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_KEY}&client_secret={SECRET_KEY}'
  resp = http.post(url, headers=HEADERS)
  data: Dict = resp.json()
  if 'error' in data:
    print('>> error:', data.get('error'))
    print('>> error_description:', data.get('error_description'))
    exit(-1)

  ACCESS_TOKEN = data.get('access_token')
  assert ACCESS_TOKEN, '>> missing ACCESS_TOKEN'
  print('>> ACCESS_TOKEN:', ACCESS_TOKEN)

  print('=' * 72)


def query(fp:Path, prompt:str) -> Dict:
  url = f'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/image2text/{API_EP}?access_token={ACCESS_TOKEN}'
  payload = {
    'prompt': prompt,
    'image': base64_img(fp),
  }
  try:
    resp = http.post(url, headers=HEADERS, json=payload)
    data: Dict = resp.json()
  except:
    print_exc()
    return

  if 'error_code' in data:
    print('>> error_code:', data.get('error_code'))
    print('>> error_msg:', data.get('error_msg'))
    exit(-1)

  print('>> resp:')
  pp(data)
  return data


def run(args):
  fps = list(Path(args.img_dp).iterdir())
  fps.sort(key=lambda k: int(Path(k).stem))   # keep order
  if args.limit > 0: fps = fps[:args.limit]   # limit count 

  db = RecordDB(args.db_fp)
  try:
    for fp in fps:
      ts_req = now()
      res = query(fp, QUERY_PROMPT)
      ts_res = now()

      if res is None:
        print('>> error query:', fp)
        continue

      img = Image.open(fp).convert('RGB')
      db.add(prompt, img, res, ts_req, ts_res)
  except KeyboardInterrupt:
    print('>> Exit by Ctrl+C')
  except:
    print_exc()
  finally:
    db.close()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-I', '--img_dp', required=True, type=Path, help='image folder')
  parser.add_argument('-L', '--limit', default=-1, type=int, help='limit input image count')
  parser.add_argument('-O', '--db_fp', default=LOG_PATH / 'query_api.sqlite3', type=Path, help='query record database file')
  parser.add_argument('-K', '--key_fp', default='API_KEY.yaml', type=Path, help='credential key file (*.yaml)')
  args = parser.parse_args()

  setup(args.key_fp)

  run(args)
