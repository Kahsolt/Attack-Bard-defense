"""
all the implementation here, the range of images values is [0, 1]
if a model need normalization, please adding the normalization part in the model, not in loader or attacker

"""
from argparse import ArgumentParser

from defenses import get_dfn
from .AdversarialInput import *


ATTACKS = {
  'none': nn.Identity,
  #'ssa': SpectrumSimulationAttack,
  'ssa_cw': SSA_CommonWeakness,
  #'pi': PI_FGSM,
  #'naattack': NAttack,
  #'svre': MI_SVRE,
  'di': DI_MI_FGSM,
  #'sgd': SGD,
  #'ti': MI_TI_FGSM,
  #'vmi_fgsm': VMI_FGSM,
  #'vmi_in_com': VMI_Inner_CommonWeakness,
  #'vmi_out_com': VMI_Outer_CommonWeakness,
  #'bim': BIM,
  #'fgsm': FGSM,
  'pgd': PGD,
  'mi': MI_FGSM,
  #'mi_rap': MI_RAP,
  #'mi_sam': MI_SAM,
  #'mi_cse': MI_CosineSimilarityEncourager,
  #'mi_rw': MI_RandomWeight,
  #'mi_cw': MI_CommonWeakness,
  #'adam_cw': Adam_CommonWeakness,
}


def get_atk(models, criterion):
  parser = ArgumentParser()
  parser.add_argument('--atk',        '-atk',  required=True, choices=ATTACKS.keys())
  parser.add_argument('--epsilon',    '-eps',  type=eval,     default=16/255)
  parser.add_argument('--step_size',  '-size', type=eval,     default=1/255)
  parser.add_argument('--total_step', '-step', type=int,      default=100)
  args, _ = parser.parse_known_args()

  dfn = get_dfn()
  return ATTACKS[args.atk](models, dfn, epsilon=args.epsilon, step_size=args.step_size, total_step=args.total_step, criterion=criterion)
