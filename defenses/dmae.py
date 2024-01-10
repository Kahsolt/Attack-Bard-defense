#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/10 

from defenses.mae import hijack_mae
from defenses.utils import *
DMAE_PATH = REPO_PATH / 'dmae'
register_path(DMAE_PATH)
from models_dmae import dmae_vit_base_patch16, dmae_vit_large_patch16, DenoisingMaskedAutoencoderViT

parser = ArgumentParser()
# ↓↓↓ You can tune these ↓↓↓
parser.add_argument('--model_name', type=str, default='base', choices=['base', 'large'])
args, _ = parser.parse_known_args()

if 'model config':
  if args.model_name == 'base':
    ckpt_name = 'dmae_base_sigma_0.25_mask_0.75_1100e'
  elif args.model_name == 'large':
    ckpt_name = 'dmae_large_sigma_0.25_mask_0.75_1600e'
  model_path = DMAE_PATH / 'models' / f'{ckpt_name}.pth'


class DMAE_dfn:

  def __init__(self):
    if args.model_name == 'base':
      mae = dmae_vit_base_patch16()
    elif args.model_name == 'large':
      mae = dmae_vit_large_patch16()
    mae.load_state_dict(torch.load(model_path, map_location=device))
    mae = mae.eval().to(device)
    self.model: DenoisingMaskedAutoencoderViT = hijack_mae(mae)

  def __call__(self, x:Tensor, n_split:int=8) -> Tensor:
    return self.model.cross_infer(x, n_split)


if __name__ == '__main__':
  dfn = DMAE_dfn()
  x = torch.rand([1, 3, 224, 224]).to(device)
  print('x.shape:', x.shape)
  y = dfn(x)
  print('y.shape:', y.shape)
  d = (x - y).abs().mean().item()
  print('err:', d)
