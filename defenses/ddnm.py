#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/30 

from defenses.utils import *
DDNM_PATH = REPO_PATH / 'DDNM'
register_path(DDNM_PATH)

import yaml
from tqdm import tqdm

from main import dict2namespace
from guided_diffusion.script_util import create_model
from guided_diffusion.diffusion import Diffusion, get_schedule_jump, compute_alpha, inverse_data_transform, MeanUpsample
from functions.ckpt_util import download

parser = ArgumentParser()
# ↓↓↓ You can tune these ↓↓↓
parser.add_argument("--deg", type=str, default='sr_averagepooling', choices=['denoising', 'sr_averagepooling'], help="Degradation")
parser.add_argument("--steps", type=int, default=4, help="diffusion steps")
parser.add_argument("--sigma_y", type=float, default=0.1, help="sigma_y")
parser.add_argument("--eta", type=float, default=0.85, help="Eta")
# ↓↓↓ DO NOT TOUCH ↓↓↓
parser.add_argument("--config", type=str, default='imagenet_256.yml', help="Path to the config file")
parser.add_argument("--exp", type=str, default=(DDNM_PATH / "exp"), help="Path for saving running related data.")
parser.add_argument("-n", "--noise_type", type=str, default="gaussian", help="gaussian | 3d_gaussian | poisson | speckle")
parser.add_argument("--deg_scale", type=float, default=1.0, help="deg_scale")
parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
args, _ = parser.parse_known_args()

if 'model config':
  with open(DDNM_PATH / "configs" / args.config, "r") as f:
    config = yaml.safe_load(f)
  config = dict2namespace(config)
  config.device = device
  # override diffusion sample steps
  config.time_travel.T_sampling = args.steps

  assert config.model.type == 'openai'
  assert not config.model.class_cond


def setup_model(self:Diffusion):
  config_dict = vars(self.config.model)
  model = create_model(**config_dict)
  if self.config.model.use_fp16:
    model.convert_to_fp16()
  ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
  if not os.path.exists(ckpt):
    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
  model.load_state_dict(torch.load(ckpt, map_location=self.device))
  model.eval().to(self.device)
  model = torch.nn.DataParallel(model)
  return model

def simplified_ddnm_plus_hijack(self:Diffusion, model:Module, y:Tensor) -> Tensor:
  g = torch.Generator()
  g.manual_seed(args.seed)

  # get degradation operator
  if args.deg == 'denoising':
    A = lambda z: z
    Ap = A
  elif args.deg == 'sr_averagepooling':
    scale = round(args.deg_scale)
    A = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
    Ap = lambda z: MeanUpsample(z,scale)

  args.sigma_y = 2 * args.sigma_y # to account for scaling to [-1,1]
  sigma_y = args.sigma_y

  # init x_T
  x = torch.randn(
    y.shape[0],
    config.data.channels,
    config.data.image_size,
    config.data.image_size,
    device=self.device,
  )

  skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
  n = x.size(0)
  xs = x    # save last step xi

  times = get_schedule_jump(config.time_travel.T_sampling, config.time_travel.travel_length, config.time_travel.travel_repeat)
  time_pairs = list(zip(times[:-1], times[1:]))

  # reverse diffusion sampling
  for i, j in tqdm(time_pairs):
    i, j = i*skip, j*skip
    if j < 0: j = -1
    assert j < i  # normal sampling

    t = (torch.ones(n) * i).to(x.device)
    next_t = (torch.ones(n) * j).to(x.device)
    at = compute_alpha(self.betas, t.long())
    at_next = compute_alpha(self.betas, next_t.long())
    sigma_t = (1 - at_next**2).sqrt()
    xt = xs
    et = model(xs, t)
    if et.size(1) == 6:
      et = et[:, :3]
    # Eq. 12
    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
    # Eq. 19
    if sigma_t >= at_next*sigma_y:
      lambda_t = 1.
      gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
    else:
      lambda_t = (sigma_t)/(at_next*sigma_y)
      gamma_t = 0.
    # Eq. 17
    x0_t_hat = x0_t - lambda_t*Ap(A(x0_t) - y)
    eta = self.args.eta
    c1 = (1 - at_next).sqrt() * eta
    c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
    # different from the paper, we use DDIM here instead of DDPM
    xs = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

  x = inverse_data_transform(config, xs)
  return x


class DDNM_dfn:

  def __init__(self):
    self.runner = Diffusion(args, config, device)
    self.model = setup_model(self.runner)

  @torch.enable_grad()
  def __call__(self, x:Tensor) -> Tensor:
    B, C, W, H = x.shape
    x = F.interpolate(x, (256, 256), mode='bilinear')
    x = simplified_ddnm_plus_hijack(self.runner, self.model, x)
    x = F.interpolate(x, (H, W), mode='bilinear')
    return x


if __name__ == '__main__':
  dfn = DDNM_dfn()
  x = torch.rand([1, 3, 224, 224]).to(device)
  print('x.shape:', x.shape)
  y = dfn(x)
  print('y.shape:', y.shape)
