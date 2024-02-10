#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/2/10 

import numpy as np

from defenses.utils import *
COMDEFEND_PATH = REPO_PATH / 'Comdefend'
#register_path(COMDEFEND_PATH)

if 'model config':
  model_enc_path = COMDEFEND_PATH / 'checkpoints' / 'enc20_0.0001.npy'
  model_dec_path = COMDEFEND_PATH / 'checkpoints' / 'dec20_0.0001.npy'


class encoder(nn.Module):
    # initializers
    def __init__(self, d=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 12, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.features(x)


class decoder(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.features(x)


class ComDefend_dfn:

  def __init__(self):
    enc = encoder()
    for v in enc.parameters():
      v.requires_grad = False
    ckpt = np.load(model_enc_path)
    self.enc = enc.eval().to(device)

    dec = decoder()
    for v in dec.parameters():
      v.requires_grad = False
    ckpt = np.load(model_dec_path)
    self.dec = dec.eval().to(device)

  @torch.enable_grad()
  def __call__(self, x:Tensor) -> Tensor:
    return self.dec(self.enc(x))


if __name__ == '__main__':
  dfn = ComDefend_dfn()
  x = torch.rand([1, 3, 224, 224]).to(device)
  print('x.shape:', x.shape)
  y = dfn(x)
  print('y.shape:', y.shape)
  d = (x - y).abs().mean().item()
  print('err:', d)
