''' input range is [0, 1] in float32 '''

from .blur_and_sharpen import BlurAndSharpen_dfn, BlurAndSharpenMy_dfn
from .realesrgan import RealESRGAN_dfn
from .scunet import SCUNet_dfn
from .ddnm import DDNM_dfn
from .mae import MAE_dfn
from .dmae import DMAE_dfn
from .comdefend import ComDefend_dfn
from .jpeg import JPEG_dfn
from .local_grad_smooth import LGS_dfn

from .utils import *


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
