#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/30 

import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.nn import Module

BASE_PATH = Path(__name__).parent
REPO_PATH = BASE_PATH / 'repo'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def register_path(dp:Path):
  assert dp.is_dir()
  sys.path.append(str(dp))
