#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/2/10 

from defenses.utils import *
LGS_PATH = REPO_PATH / 'local_gradients_smoothing'
register_path(LGS_PATH)
from lgs import LocalGradientsSmoothing
from lgs import Gradient, GradientSmooth


def Gradient_forward_hijack(self:Gradient, img:Tensor):
  batch_size = img.shape[0]
  img_aux = img.reshape(-1, img.shape[-2], img.shape[-1]).unsqueeze(1)
  grad_x = self.d_x(img_aux)
  grad_y = self.d_y(img_aux)
  grad = self.zero_pad_x(grad_x).pow(2) + self.zero_pad_y(grad_y).pow(2)
  grad = grad.sqrt()
  grad = grad.squeeze(1).reshape(batch_size, -1, img_aux.shape[-2], img_aux.shape[-1])
  return grad

Gradient.forward = Gradient_forward_hijack


def GradientSmooth_forward_hijack(self:GradientSmooth, img:Tensor):
  batch_size = img.shape[0]
  img_aux = img.reshape(-1, img.shape[-2], img.shape[-1]).unsqueeze(1)
  grad = self.d_x(img_aux).pow(2) + self.d_y(img_aux).pow(2)
  grad = grad.sqrt()
  grad = grad.squeeze(1).reshape(batch_size, -1, img_aux.shape[-2], img_aux.shape[-1])
  return grad

GradientSmooth.forward = GradientSmooth_forward_hijack


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
