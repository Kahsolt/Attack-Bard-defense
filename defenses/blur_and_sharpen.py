#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/07 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor


class BlurAndSharpen:

  def __init__(self, k:int=5, s:float=1e-5):
    self.k = k
    self.s = s

  def __call__(self, x:Tensor) -> Tensor:
    # k=3 kernel: [
    #   [0.0571, 0.1248, 0.0571],
    #   [0.1248, 0.2725, 0.1248],
    #   [0.0571, 0.1248, 0.0571]
    # ]
    x = TF.gaussian_blur(x, kernel_size=self.k)
    # uses SMOOTH filter: [
    #   [1, 1, 1],
    #   [1, 5, 1],
    #   [1, 1, 1]
    # ]
    x = TF.adjust_sharpness(x, sharpness_factor=self.s)
    return x


class BlurAndSharpenMy:

  def __init__(self):
    # https://blog.csdn.net/lphbtm/article/details/18001933
    self.blur = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3, padding=3//2, bias=False, padding_mode='replicate')
    kernel = Tensor([
      [1, 2, 1],
      [2, 4, 2],
      [1, 2, 1],
    ]).unsqueeze_(0).unsqueeze_(0).expand(3, -1, -1, -1) / 16
    self.blur.weight.data = nn.Parameter(kernel, requires_grad=False)
    self.blur.requires_grad_(False)

    # ref: https://en.wikipedia.org/wiki/Unsharp_masking#Digital_unsharp_masking
    self.sharpen = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3, padding=3//2, bias=False, padding_mode='replicate')
    kernel = Tensor([
      [ 0, -1,  0],
      [-1,  5, -1],
      [ 0, -1,  0],
    ]).unsqueeze_(0).unsqueeze_(0).expand(3, -1, -1, -1)
    self.sharpen.weight.data = nn.Parameter(kernel, requires_grad=False)
    self.sharpen.requires_grad_(False)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.blur(x)
    x = self.sharpen(x)
    return x


if __name__ == '__main__':
  import numpy as np
  from PIL import Image
  import matplotlib.pyplot as plt

  idx = 0
  fp_raw = f'./dataset/NIPS17/{idx}.png'
  fp_adv = f'./dataset/ssa-cwa-200/{idx}.png'
  im_raw = np.asarray(Image.open(fp_raw).convert('RGB'), dtype=np.float32) / 255.0
  im_adv = np.asarray(Image.open(fp_adv).convert('RGB'), dtype=np.float32) / 255.0
  X  = torch.from_numpy(im_raw).permute([2, 0, 1]).unsqueeze_(0)
  AX = torch.from_numpy(im_adv).permute([2, 0, 1]).unsqueeze_(0)

  dfn = BlurAndSharpen()
  with torch.no_grad():
    X_dfn:  Tensor = dfn(X)
    AX_dfn: Tensor = dfn(AX)

  im_raw_dfn =  X_dfn.squeeze_(0).permute([1, 2, 0]).clamp_(0.0, 1.0).cpu().numpy()
  im_adv_dfn = AX_dfn.squeeze_(0).permute([1, 2, 0]).clamp_(0.0, 1.0).cpu().numpy()

  L1_nodfn = np.abs(im_adv - im_raw)
  print('|AX - X|')
  print('   Linf:', L1_nodfn.max())
  print('   L1  :', L1_nodfn.mean())
  L1_dfn = np.abs(im_adv_dfn - im_raw_dfn)
  print('|dfn(AX) - dfn(X)|')
  print('  Linf:', L1_dfn.max())
  print('  L1  :', L1_dfn.mean())

  plt.subplot(221) ; plt.imshow(im_raw)     ; plt.axis('off') ; plt.title('X')
  plt.subplot(222) ; plt.imshow(im_adv)     ; plt.axis('off') ; plt.title('AX')
  plt.subplot(223) ; plt.imshow(im_raw_dfn) ; plt.axis('off') ; plt.title('dfn(X)')
  plt.subplot(224) ; plt.imshow(im_adv_dfn) ; plt.axis('off') ; plt.title('dfn(AX)')
  plt.tight_layout()
  plt.show()
