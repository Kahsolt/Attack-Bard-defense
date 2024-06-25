#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/10

# 客观度量: 模型输入图像的相似度 (正常样本 vs 对抗样本)

import os
import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, normalized_root_mse, peak_signal_noise_ratio, structural_similarity

from utils.img_proc import *
from utils.niqe.niqe import get_niqe


def plot_cmp(img_raw:PILImage, img_adv:PILImage):
  img_raw_lo = img_raw.filter(ImageFilter.GaussianBlur(3))
  img_adv_lo = img_adv.filter(ImageFilter.GaussianBlur(3))
  im_raw_lo = pil_to_npimg(img_raw_lo)
  im_adv_lo = pil_to_npimg(img_adv_lo)
  im_raw = pil_to_npimg(img_raw)
  im_adv = pil_to_npimg(img_adv)
  im_raw_hi = minmax_norm(npimg_diff(im_raw, im_raw_lo))
  im_adv_hi = minmax_norm(npimg_diff(im_adv, im_adv_lo))

  print('[metrics]')
  niqe_raw = get_niqe(im_raw) ; print('  niqe_raw:', niqe_raw)
  niqe_adv = get_niqe(im_adv) ; print('  niqe_adv:', niqe_adv)
  mse  = mean_squared_error     (im_raw, im_adv) ; print('  mse:', mse)
  rmse = normalized_root_mse    (im_raw, im_adv) ; print('  rmse:', rmse)
  psnr = peak_signal_noise_ratio(im_raw, im_adv) ; print('  psnr:', psnr)
  ssim = structural_similarity  (to_gray(im_raw), to_gray(im_adv)) ; print('  ssim:', ssim)

  dx    = npimg_abs_diff(im_adv,    im_raw,    name='dx')
  dx_lo = npimg_abs_diff(im_adv_lo, im_raw_lo, name='dx_lo')
  dx_hi = npimg_abs_diff(im_adv_hi, im_raw_hi, name='dx_hi')
  dx0    = minmax_norm(dx, vmin=0, vmax=16)   # eps=16/255
  dx0_lo = minmax_norm(dx_lo)
  dx0_hi = minmax_norm(dx_hi)
  dx1    = minmax_norm(to_ch_avg(dx), vmin=0, vmax=16)
  dx1_lo = minmax_norm(to_ch_avg(dx_lo))
  dx1_hi = minmax_norm(to_ch_avg(dx_hi))

  plt.clf()
  plt.subplot(3, 4,  1) ; plt.axis('off') ; plt.title('X')      ; plt.imshow(im_raw)
  plt.subplot(3, 4,  2) ; plt.axis('off') ; plt.title('AX')     ; plt.imshow(im_adv)
  plt.subplot(3, 4,  3) ; plt.axis('off') ; plt.title('DX3')    ; plt.imshow(dx0)
  plt.subplot(3, 4,  4) ; plt.axis('off') ; plt.title('DX1')    ; plt.imshow(dx1)
  plt.subplot(3, 4,  5) ; plt.axis('off') ; plt.title('X_lo')   ; plt.imshow(im_raw_lo)
  plt.subplot(3, 4,  6) ; plt.axis('off') ; plt.title('AX_lo')  ; plt.imshow(im_adv_lo)
  plt.subplot(3, 4,  7) ; plt.axis('off') ; plt.title('DX3_lo') ; plt.imshow(dx0_lo)
  plt.subplot(3, 4,  8) ; plt.axis('off') ; plt.title('DX1_lo') ; plt.imshow(dx1_lo)
  plt.subplot(3, 4,  9) ; plt.axis('off') ; plt.title('X_hi')   ; plt.imshow(im_raw_hi)
  plt.subplot(3, 4, 10) ; plt.axis('off') ; plt.title('AX_hi')  ; plt.imshow(im_adv_hi)
  plt.subplot(3, 4, 11) ; plt.axis('off') ; plt.title('DX3_hi') ; plt.imshow(dx0_hi)
  plt.subplot(3, 4, 12) ; plt.axis('off') ; plt.title('DX1_hi') ; plt.imshow(dx1_hi)
  plt.show()


def stats_cmp(img_raw:PILImage, img_adv:PILImage) -> Any:
  im_raw = pil_to_npimg(img_raw)
  im_adv = pil_to_npimg(img_adv)

  niqe_raw = get_niqe(im_raw)
  niqe_adv = get_niqe(im_adv)
  mse  = mean_squared_error     (im_raw, im_adv)
  rmse = normalized_root_mse    (im_raw, im_adv)
  psnr = peak_signal_noise_ratio(im_raw, im_adv)
  ssim = structural_similarity  (to_gray(im_raw), to_gray(im_adv))

  im_raw_f32 = im_raw.astype(np.float32) / 255.0
  im_adv_f32 = im_adv.astype(np.float32) / 255.0

  dx = np.abs(im_adv_f32 - im_raw_f32)
  Linf = dx.max()
  L1 = dx.mean()
  L2 = np.linalg.norm(dx.flatten(), ord=2)

  return {
    'Linf': Linf.item(),
    'L1': L1.item(),
    'L2': L2.item(),
    'mse': mse,
    'rmse': rmse,
    'psnr': psnr,
    'ssim': ssim,
    'niqe+': niqe_adv - niqe_raw,   # diff value
    'niqe_raw': niqe_raw,
    'niqe_adv': niqe_adv,
  }


def run(args):
  dp_raw: Path = args.input
  dp_adv: Path = args.output

  adv_fps = list(dp_adv.iterdir())    # keep numerical order
  adv_fps.sort(key=img_fps_sort_fn)

  results = {}
  for fp_adv in tqdm(adv_fps):
    fp_raw = dp_raw / fp_adv.name

    img_raw = load_img(fp_raw)
    img_adv = load_img(fp_adv)
    #plot_cmp(img_raw, img_adv)
    stats = stats_cmp(img_raw, img_adv)
    results[fp_adv.name] = stats

  stats = {}
  metric_names = list(list(results.values())[0].keys())
  for metric in metric_names:
    data = np.asarray([v[metric] for v in results.values()])
    stats[metric] = {
      'avg': data.mean(),
      'std': data.std(),
      'max': data.max(),
      'min': data.min(),
    }

  out_dp: Path = LOG_PATH / Path(__file__).stem
  out_dp.mkdir(exist_ok=True)
  save_fp = out_dp / f'{str(args.output).replace(os.sep, ".")}.json'
  print(f'>> save to {save_fp}')
  with open(save_fp, 'w', encoding='utf-8') as fh:
    data = {
      'stats': stats,
      'results': results,
    }
    json.dump(data, fh, ensure_ascii=False, indent=2)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-I', '--input',  type=Path, default=DATA_RAW_PATH, help='clean image folder')
  parser.add_argument('-O', '--output', type=Path, default=DATA_ADV_PATH, help='adv image folder')
  args = parser.parse_args()

  run(args)
