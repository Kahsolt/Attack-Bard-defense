import os

import torch
from torchvision import transforms
from tqdm import tqdm

from attacks import *
from defenses import get_cmdargs
from surrogates import (
    BlipFeatureExtractor,
    ClipFeatureExtractor,
    EnsembleFeatureLoss,
    VisionTransformerFeatureExtractor,
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
def ssa_cw_count_to_index(count, num_models=len(models), ssa_N=20):
    max = ssa_N * num_models
    count = count % max
    count = count // ssa_N
    return count
criterion = EnsembleFeatureLoss(models, ssa_cw_count_to_index, feature_loss=torch.nn.MSELoss())
attacker = get_atk(models, criterion)


''' Go!! '''
dir = args.output or "./attack_img_encoder_misdescription/"
if not os.path.exists(dir):
    os.mkdir(dir)
id = 0
for i, x in enumerate(tqdm(images)):
    print(f'>> img-{i}')
    x = x.cuda()
    criterion.set_ground_truth(x)
    adv_x = attacker(x, None)
    save_list_images(adv_x, dir, begin_id=id)
    id += x.shape[0]
