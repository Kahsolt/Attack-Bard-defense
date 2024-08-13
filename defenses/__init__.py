''' input range is [0, 1] in float32 '''

try:
  from .blur_and_sharpen import BlurAndSharpen_dfn, BlurAndSharpenMy_dfn
except ImportError:
  BlurAndSharpen_dfn = None
  BlurAndSharpenMy_dfn = None
  print('>> WARN: blur_and_sharpen not available')
try:
  from .realesrgan import RealESRGAN_dfn
except ImportError:
  RealESRGAN_dfn = None
  print('>> WARN: realesrgan not available')
try:
  from .scunet import SCUNet_dfn
except:
  SCUNet_dfn = None
  print('>> WARN: scunet not available')
try:
  from .ddnm import DDNM_dfn
except:
  DDNM_dfn = None
  print('>> WARN: ddnm not available')
try:
  from .mae import MAE_dfn
except:
  MAE_dfn = None
  print('>> WARN: mae not available')
try:
  from .dmae import DMAE_dfn
except:
  DMAE_dfn = None
  print('>> WARN: dmae not available')
try:
  from .comdefend import ComDefend_dfn
except:
  ComDefend_dfn = None
  print('>> WARN: comdefend not available')
try:
  from .jpeg import JPEG_dfn
except:
  JPEG_dfn = None
  print('>> WARN: jpeg not available')
try:
  from .local_grad_smooth import LGS_dfn
except:
  LGS_dfn = None
  print('>> WARN: local_grad_smooth not available')

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
  parser.add_argument('-dfn', '--dfn', default='none', choices=DEFENSES.keys())
  args, _ = parser.parse_known_args()
  return DEFENSES[args.dfn]()


def get_cmdargs() -> int:    # FIXME: this shouldn't be put here
  parser = ArgumentParser()
  parser.add_argument('-L', '--limit', default=100, type=int)
  parser.add_argument('-O', '--output')
  args, _ = parser.parse_known_args()
  return args
