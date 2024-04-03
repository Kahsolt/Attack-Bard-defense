#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

# misc utils for image processing or anything common :(

import warnings ; warnings.filterwarnings('ignore', category=UserWarning)

from pathlib import Path
import hashlib
from PIL import Image, ImageFilter
from PIL.Image import Image as PILImage
from datetime import datetime
from typing import *

import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from numpy.typing import NDArray
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_PATH = Path(__file__).parent.parent
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)
DATA_PATH = BASE_PATH / 'dataset'
DATA_RAW_PATH = DATA_PATH / 'NIPS17'
DATA_ADV_PATH = DATA_PATH / 'ssa-cwa-200' # pregen

IM_U8_TYPES = ['u', 'u8', 'uint8', np.uint8]
IM_F32_TYPES = ['f', 'f32', 'float32', np.float32]
IM_TYPES = IM_U8_TYPES + IM_F32_TYPES

npimg_u8 = NDArray[np.uint8]        # vrng [0, 255]
npimg_f32 = NDArray[np.float32]     # vrng [0, 1]
npimg = Union[npimg_u8, npimg_f32]
npimg_dx = NDArray[np.int16]        # vrng [-255, 255]
npimg_hi = NDArray[np.float32]      # vrng [-1, 1]

mean = lambda x: sum(x) / len(x) if len(x) else 0.0
img_fps_sort_fn = lambda k: int(Path(k).stem)


def now() -> int:
  return int(datetime.now().timestamp())


def read_file(fp:Path) -> bytes:
  with open(fp, 'rb') as fh:
    bytedata = fh.read()
  return bytedata

def hash_bdata(bdata:bytes, method='sha512') -> str:
  assert isinstance(bdata, bytes)
  hash_fn: Callable[[bytes], hashlib._Hash] = getattr(hashlib, method)
  return hash_fn(bdata).hexdigest()


def load_img(fp:Path) -> PILImage:
  return Image.open(fp).convert('RGB')

def load_im(fp:Path, dtype:str='u8') -> npimg:
  return pil_to_npimg(load_img(fp), dtype)

def pil_to_npimg(img:PILImage, dtype:str='u8') -> npimg:
  assert dtype in IM_TYPES
  im = np.asarray(img, dtype=np.uint8)
  if dtype in IM_U8_TYPES: return im
  return im.astype(np.float32) / 255.0

def npimg_to_pil(im:npimg) -> PILImage:
  assert im.dtype in IM_TYPES
  if im.dtype in IM_F32_TYPES:
    assert 0.0 <= im.min() and im.max() <= 1.0
  return Image.fromarray(im)

def hwc2chw(im:npimg) -> npimg:
  return im.transpose(2, 0, 1)

def chw2hwc(im:npimg) -> npimg:
  return im.transpose(1, 2, 0)

def npimg_to_tensor(im:npimg_f32) -> Tensor:
  return torch.from_numpy(hwc2chw(im))

def to_gray(im:npimg) -> npimg:
  return pil_to_npimg(npimg_to_pil(im).convert('L'))

def to_ch_avg(x:ndarray) -> ndarray:
  return np.tile(x.mean(axis=-1, keepdims=True), (1, 1, 3))

def npimg_abs_diff(x:npimg, y:npimg, name:str=None) -> npimg:
  d: ndarray = np.abs(npimg_diff(x, y))
  if name:
    print(f'[{name}]')
    print('  Linf:', d.max() / 255)
    print('  L1:',  d.mean() / 255)
  return d

def npimg_diff(x:npimg_u8, y:npimg_u8) -> npimg_dx:
  return x.astype(np.int16) - y.astype(np.int16)

def minmax_norm(dx:npimg_dx, vmin:int=None, vmax:int=None) -> npimg_u8:
  vmin = vmin or dx.min()
  vmax = vmax or dx.max()
  out = (dx - vmin) / (vmax - vmin)
  return (out * 255).astype(np.uint8)

def Linf_L1_L2(X:Tensor, AX:Tensor=None) -> Tuple[float, float, float]:
  if AX is None:
    DX = X
  else:
    DX = (AX - X).abs()
  Linf = DX.max()
  L1 = DX.mean()
  L2 = (DX**2).sum().sqrt()
  return [x.item() for x in [Linf, L1, L2]]
