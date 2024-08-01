import torch
from surrogates import (
    BlipFeatureExtractor,
    ClipFeatureExtractor,
    EnsembleFeatureLoss,
    VisionTransformerFeatureExtractor,
)
from utils import get_list_image, save_list_images
from tqdm import tqdm
#from attacks import SpectrumSimulationAttack, SSA_CommonWeakness
from attacks import *
from torchvision import transforms
import os

from defenses import get_dfn, get_cmdargs
args = get_cmdargs()

images = get_list_image("./dataset/NIPS17", limit=args.limit)
resizer = transforms.Resize((224, 224))
images = [resizer(i).unsqueeze(0) for i in images]


class Model:
    def __init__(self, name):
        self.name = name

class ModelContainer:
    def __init__(self, models):
        self.models = models

    def __iter__(self):
        return iter(self.models)

# 创建一些模型实例
model1 = Model("Model 1")
model2 = Model("Model 2")
model3 = Model("Model 3")

# 创建一个模型容器实例
container = ModelContainer([model1, model2, model3])
    
attacker = get_atk(model=container, criterion=None)

breakpoint()