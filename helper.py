#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/30 

from argparse import ArgumentParser

from defenses import *

DEFENSES = {
  'none': nn.Identity,
  'blur_and_sharpen': BlurAndSharpen_dfn,
  'realesrgan': RealESRGAN_dfn,
  'scunet': SCUNet_dfn,
  'ddnm': DDNM_dfn,
  'mae': MAE_dfn,
  'dmae': DMAE_dfn,
  'comdef': ComDefend_dfn,
  'jpeg': JPEG_dfn,
  'lgs': LGS_dfn,
}


def get_dfn():
  parser = ArgumentParser()
  parser.add_argument('--dfn', required=True, choices=DEFENSES.keys())
  args, _ = parser.parse_known_args()
  return DEFENSES[args.dfn]()
