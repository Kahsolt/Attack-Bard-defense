#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/03

# 客观度量: 模型输出文本的相似度 (对正常样本的描述 vs 对对抗样本的描述)

import json
from argparse import ArgumentParser

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_dot_score, pairwise_cos_sim, pairwise_angle_sim

from query_api import QUERY_PROMPTS
from utils.record_db import *
from utils.img_proc import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


PRETRAINED_CHECKPOINTS = [
  # official: https://www.sbert.net/docs/pretrained_models.html#model-overview
  'all-mpnet-base-v2',
  'multi-qa-mpnet-base-dot-v1',
  'all-distilroberta-v1',
  'all-MiniLM-L12-v2',
  'multi-qa-distilbert-cos-v1',
  'all-MiniLM-L6-v2',
  'multi-qa-MiniLM-L6-cos-v1',
  'paraphrase-multilingual-mpnet-base-v2',
  'paraphrase-albert-small-v2',
  'paraphrase-multilingual-MiniLM-L12-v2',
  'paraphrase-MiniLM-L3-v2',
  'distiluse-base-multilingual-cased-v1',
  'distiluse-base-multilingual-cased-v2',
  # contrib
  'amu/tao-k',
  'BAAI/bge-m3',
  'aspire/acge_text_embedding',
  'mixedbread-ai/mxbai-embed-large-v1',
]


def run(args):
  ref_fps = list(Path(args.img_dp).iterdir())
  ref_fps.sort(key=img_fps_sort_fn)
  adv_fps = list(Path(args.adv_dp).iterdir())
  adv_fps.sort(key=img_fps_sort_fn)

  db = RecordDB(args.db_fp)
  model = SentenceTransformer(args.model, device=device)
  pid = db.get_prompt_id(QUERY_PROMPTS[args.pid])

  dot_scores, cos_sims, angle_sims = [], [], []
  total, ok, not_found = 0, 0, 0
  for adv_fp, ref_fp in zip(adv_fps, ref_fps):
    total += 1

    try:
      iid_adv = db.get_image_id(adv_fp)
      rec = db.get(pid, iid_adv)[0]
      sent_adv = json.loads(rec)['result']
    except:
      not_found += 1
      continue
    try:
      iid_ref = db.get_image_id(ref_fp)
      rec = db.get(pid, iid_ref)[0]
      sent_ref = json.loads(rec)['result']
    except:
      not_found += 1
      continue

    embed_adv, embed_ref = model.encode([sent_adv, sent_ref], convert_to_tensor=True)
    dot_scores.append(pairwise_dot_score(embed_adv, embed_ref).item())
    cos_sims  .append(pairwise_cos_sim  (embed_adv, embed_ref).item())
    angle_sims.append(pairwise_angle_sim(embed_adv, embed_ref).item())

    ok += 1

  print(f'>> total: {total}, ok: {ok}, not found: {not_found}')
  print('dot score:', mean(dot_scores))
  print('cos sim:',   mean(cos_sims))
  print('angle sim:', mean(angle_sims))
  ok, not_found = 0, 0


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M',  '--model',   default='all-MiniLM-L6-v2', choices=PRETRAINED_CHECKPOINTS, help='model ckpt')
  parser.add_argument('-X',  '--img_dp',  type=Path, default=DATA_RAW_PATH, help='clean image folder as reference')
  parser.add_argument('-AX', '--adv_dp',  type=Path, required=True, help='adversarial image folder')
  parser.add_argument('-P',  '--pid',     default=0, type=int, help='query prompt id in the predefined list')
  parser.add_argument('-D',  '--db_fp',   type=Path, default=DB_FILE, help='query record database file')
  parser.add_argument('-S',  '--save_fp', type=Path, default=(LOG_PATH / f'{Path(__file__).name}.json'), help='result output file')
  args = parser.parse_args()

  run(args)
