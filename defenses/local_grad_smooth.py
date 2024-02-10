#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/2/10 

from defenses.utils import *
LGS_PATH = REPO_PATH / 'local_gradients_smoothing'
register_path(LGS_PATH)
from lgs import LocalGradientsSmoothing


class LGS_dfn:

  def __init__(self):
    lgs = LocalGradientsSmoothing(
      smoothing_factor=2.3,
      window_size=15,
      overlap=5,
      threshold=0.1,
      grad_method='Gradient',
    )
    self.model = lgs.to(device)

  @torch.enable_grad()
  def __call__(self, x:Tensor) -> Tensor:
    return torch.cat([self.model(ch.unsqueeze(1)) for ch in torch.split(x, 1, dim=1)], dim=1)


if __name__ == '__main__':
  dfn = LGS_dfn()
  x = torch.rand([1, 3, 224, 224]).to(device)
  print('x.shape:', x.shape)
  y = dfn(x)
  print('y.shape:', y.shape)
  d = (x - y).abs().mean().item()
  print('err:', d)
