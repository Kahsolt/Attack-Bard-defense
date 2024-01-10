#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/30 

from defenses.utils import *
SCUNET_PATH = REPO_PATH / 'SCUNet'
register_path(SCUNET_PATH)
from models.network_scunet import SCUNet

parser = ArgumentParser()
# ↓↓↓ You can tune these ↓↓↓
parser.add_argument('--model_name', type=str, default='scunet_color_real_psnr', choices=['scunet_color_real_psnr', 'scunet_color_real_gan'])
args, _ = parser.parse_known_args()

if 'model config':
  model_path = SCUNET_PATH / 'model_zoo' / f'{args.model_name}.pth'


class SCUNet_dfn:

  def __init__(self):
    model = SCUNet(in_nc=3, config=[4,4,4,4,4,4,4], dim=64)
    model.load_state_dict(torch.load(model_path), strict=True)
    for v in model.parameters():
      v.requires_grad = False
    self.model = model.eval().to(device)

  @torch.enable_grad()
  def __call__(self, x:Tensor) -> Tensor:
    return self.model(x)


if __name__ == '__main__':
  dfn = SCUNet_dfn()
  x = torch.rand([1, 3, 224, 224]).to(device)
  print('x.shape:', x.shape)
  y = dfn(x)
  print('y.shape:', y.shape)
  d = (x - y).abs().mean().item()
  print('err:', d)
