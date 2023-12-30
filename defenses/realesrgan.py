#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/30 

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from defenses.utils import *
RESRGAN_PATH = REPO_PATH / 'Real-ESRGAN'
register_path(RESRGAN_PATH)
from realesrgan.utils import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

parser = ArgumentParser()
# ↓↓↓ You can tune these ↓↓↓
parser.add_argument('-n', '--model_name', type=str, default='realesr-general-x4v3', help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | realesr-animevideov3 | realesr-general-x4v3'))
parser.add_argument('-dn', '--denoise_strength', type=float, default=0.5, help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. Only used for the realesr-general-x4v3 model'))
parser.add_argument('-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')
# ↓↓↓ DO NOT TOUCH ↓↓↓
parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
args, _ = parser.parse_known_args()

if 'model config':
  # determine models according to model names
  args.model_name = args.model_name.split('.')[0]
  if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
  elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
  elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
  elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    netscale = 2
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
  elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
  elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    netscale = 4
    file_url = [
      'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
      'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
    ]

  # determine model paths
  model_path = str(RESRGAN_PATH / 'weights' / f'{args.model_name}.pth')
  if not os.path.isfile(model_path):
    for url in file_url:
      # model_path will be updated
      model_path = load_file_from_url(url=url, model_dir=str(RESRGAN_PATH / 'weights'), progress=True, file_name=None)

  # use dni to control the denoise strength
  dni_weight = None
  if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
    wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
    model_path = [model_path, wdn_model_path]
    dni_weight = [args.denoise_strength, 1 - args.denoise_strength]


def pre_process_hijack(self:RealESRGANer, x:Tensor) -> Tensor:
  self.img = x
  # pre_pad
  if self.pre_pad != 0:
    self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
  # mod pad for divisible borders
  if self.scale == 2:
    self.mod_scale = 2
  elif self.scale == 1:
    self.mod_scale = 4
  if self.mod_scale is not None:
    self.mod_pad_h, self.mod_pad_w = 0, 0
    _, _, h, w = self.img.size()
    if (h % self.mod_scale != 0):
      self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
    if (w % self.mod_scale != 0):
      self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
    self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

def RealESRGANer_call_hijack(self:RealESRGANer, x:Tensor) -> Tensor:
  pre_process_hijack(self, x)
  self.process()
  img = self.post_process()
  return img.clamp(0, 1)


class RealESRGAN_dfn:

  def __init__(self):
    self.upsampler = RealESRGANer(
      scale=netscale,
      model_path=model_path,
      dni_weight=dni_weight,
      model=model,
      tile=args.tile,
      tile_pad=args.tile_pad,
      pre_pad=args.pre_pad,
      half=False,
      gpu_id=args.gpu_id,
    )
    for v in self.upsampler.model.parameters():
      v.requires_grad = False

  @torch.enable_grad()
  def __call__(self, x:Tensor) -> Tensor:
    B, C, H, W = x.shape
    x_high = RealESRGANer_call_hijack(self.upsampler, x)
    return F.interpolate(x_high, size=(H, W), mode='nearest')


if __name__ == '__main__':
  dfn = RealESRGAN_dfn()
  x = torch.rand([1, 3, 224, 224]).to(dfn.upsampler.device)
  print('x.shape:', x.shape)
  y = dfn(x)
  print('y.shape:', y.shape)
