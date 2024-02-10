#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/2/10 

from defenses.utils import *
from defenses.DiffJPEG.DiffJPEG import DiffJPEG

parser = ArgumentParser()
# ↓↓↓ You can tune these ↓↓↓
parser.add_argument('--quality', type=int, default=80)
args, _ = parser.parse_known_args()


class JPEG_dfn:

  def __init__(self):
    self.model = DiffJPEG(height=224, width=224, quality=args.quality)
    self.model = self.model.eval().to(device)

  @torch.enable_grad()
  def __call__(self, x:Tensor) -> Tensor:
    return self.model(x)


if __name__ == '__main__':
  dfn = JPEG_dfn()
  x = torch.rand([1, 3, 224, 224]).to(device)
  print('x.shape:', x.shape)
  y = dfn(x)
  print('y.shape:', y.shape)
  d = (x - y).abs().mean().item()
  print('err:', d)
