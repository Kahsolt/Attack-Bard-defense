import os
from time import time

import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from attacks import *
from defenses import get_cmdargs
from surrogates import (
    VisionTransformerFeatureExtractor,
    BlipFeatureExtractor,
    ClipFeatureExtractor,
    EnsembleFeatureLoss,
)
from utils import get_list_image, save_list_images


''' Data '''
args = get_cmdargs()
images = get_list_image("./dataset/NIPS17", limit=args.limit)
resizer = transforms.Resize((224, 224))
images = [resizer(i).unsqueeze(0) for i in images]


''' Model & Attacker '''
vit = VisionTransformerFeatureExtractor().eval().cuda().requires_grad_(False)
blip = BlipFeatureExtractor().eval().cuda().requires_grad_(False)
clip = ClipFeatureExtractor().eval().cuda().requires_grad_(False)
models = [vit, blip, clip]
def ssa_cw_count_to_index(count, num_models=len(models), ssa_N=1):   # NOTE: 20 for ssa-cw, 1 for others
    max = ssa_N * num_models
    count = count % max
    count = count // ssa_N
    return count
criterion = EnsembleFeatureLoss(models, ssa_cw_count_to_index, feature_loss=nn.MSELoss())
attacker = get_atk(models, criterion)


''' Go!! '''
dir = args.output or "./attack_img_encoder_misdescription/"
if not os.path.exists(dir):
    os.mkdir(dir)

id = 0
s = time()
for i, x in enumerate(tqdm(images)):
    print(f'>> img-{i}')
    x = x.cuda()
    criterion.set_ground_truth(x)
    adv_x = attacker(x, None)
    save_list_images(adv_x, dir, begin_id=id)
    id += x.shape[0]
t = time()
print(f'Done (time cost: {t - s}s = {(t - s) / 3600:.3f}h)')
