#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/10 

from defenses.utils import *
MAE_PATH = REPO_PATH / 'mae'
register_path(MAE_PATH)
from models_mae import mae_vit_large_patch16, MaskedAutoencoderViT

parser = ArgumentParser()
# ↓↓↓ You can tune these ↓↓↓
parser.add_argument('--model_name', type=str, default='mae_visualize_vit_large_ganloss', choices=['mae_visualize_vit_large_ganloss', 'mae_visualize_vit_large'])
args, _ = parser.parse_known_args()

if 'model config':
  model_path = MAE_PATH / 'models' / f'{args.model_name}.pth'


def hijack_mae(mae) -> MaskedAutoencoderViT:
  def recover_masked_one(self:MaskedAutoencoderViT, x:Tensor, x_mask:Tensor) -> Tensor:
    # NOTE: we cannot handle batch, due to possible length mismatch
    assert x.shape[0] == 1
    x_orig = x

    ''' .forward_encoder() '''
    # embed patches
    x = self.patch_embed(x)
    # add pos embed w/o cls token
    x = x + self.pos_embed[:, 1:, :]

    # masking: length -> length * mask_ratio
    def known_masking(x:Tensor, x_mask:Tensor):
      B, L, D = x.shape
      B, C, H, W = x_mask.shape
      assert C == 1 and L == H * W

      mask = x_mask.reshape((B, C*H*W))             # [B=1, L]
      ids_shuffle = torch.argsort(mask, dim=1)      # ascend: 0 is keep, 1 is remove
      ids_restore = torch.argsort(ids_shuffle, dim=1)
      len_keep = (mask[0] == 0).sum()
      ids_keep = ids_shuffle[:, :len_keep]
      # [B=1, L'<=196, D=1024]
      x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
      return x_masked, mask, ids_restore

    x, mask, ids_restore = known_masking(x, x_mask)

    # append cls token
    cls_token = self.cls_token + self.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # apply Transformer blocks
    for blk in self.blocks:
      x = blk(x)
    x = self.norm(x)

    z = self.forward_decoder(x, ids_restore)  # [N, L, p*p*3]
    y_hat = self.unpatchify(z)                # [N, C, H, W]

    # paste predicted area to known
    mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_embed.patch_size[0]**2*3)  # (N, H*W, p*p*3)
    mask = self.unpatchify(mask)              # 1 is removing, 0 is keeping
    return x_orig * (1 - mask) + y_hat * mask

  def recover_masked(self:MaskedAutoencoderViT, x:Tensor, x_mask:Tensor) -> Tensor:
    res = []
    for i in range(x.shape[0]):
      xi = recover_masked_one(self, x[i:i+1, ...], x_mask[i:i+1, ...])
      res.append(xi.cpu())
    return torch.cat(res, dim=0).to(x.device)

  mae.recover_masked = MethodType(recover_masked, mae)

  def ci_random_masking(self:MaskedAutoencoderViT, x, n_splits=4):
    """ x: [N, L, D], the patch sequence """
    assert isinstance(n_splits, int) and n_splits >= 2

    # [B=1, L=196=14*14, D=1024]
    N, L, D = x.shape
    import math
    k = math.ceil(L / n_splits)         # masked patch count

    # [B=1, L=196]
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    # [B=1, L=196], sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    # [B=1, L=196]
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # gather all non-overlapping subsets
    subsets = []
    for i in range(n_splits):
      # pick out one subset
      slicer_L = slice(None), slice(None, k*i)            # leave (k-1)/k untouched
      slicer_M = slice(None), slice(k*i, k*(i+1))         # mask 1/k patches
      slicer_R = slice(None), slice(k*(i+1), None)
      # [B=1, (k-1)/k*L]
      ids_keep = torch.cat([ids_shuffle[slicer_L], ids_shuffle[slicer_R]], axis=-1)
      #ids_keep = ids_shuffle[slicer_M]
      # [B=1, (k-1)/k*L, D=1024]
      x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
      # [B, L], generate the binary mask: 0 is keep, 1 is remove
      mask = torch.zeros([N, L], device=x.device)
      mask[slicer_M] = 1
      # unshuffle to get the binary mask
      mask = torch.gather(mask, dim=1, index=ids_restore)
      # one split
      subsets.append((x_masked, mask, k*i))

    # subsets[0][0]: [1, 171/175, 1024], use major to predict minor
    # subsets[0][1]: [1, 196], non-overlap, sum(subsets[i][1]) == 196
    # ids_restore:   [1, 196], permutation of range [0, 195]
    return subsets, ids_restore

  def ci_forward_splitter(self:MaskedAutoencoderViT, x, n_splits):
    # embed patches
    x = self.patch_embed(x)
    # add pos embed w/o cls token
    x = x + self.pos_embed[:, 1:, :]
    # masking split n overlapping subsets
    return self.ci_random_masking(x, n_splits)

  def ci_forward_encoder(self:MaskedAutoencoderViT, x):
    # append cls token
    cls_token = self.cls_token + self.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    # apply Transformer blocks
    for blk in self.blocks: x = blk(x)
    return self.norm(x)

  def ci_forward_decoder(self:MaskedAutoencoderViT, x, ids_restore, k):
    # embed tokens
    x = self.decoder_embed(x)
    # append mask tokens to sequence
    mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    x_ = x[:, 1:, :]      # no cls token
    x_ = torch.cat([x_[:, :k, :], mask_tokens, x_[:, k:, :]], dim=1)
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
    # add pos embed
    x = x + self.decoder_pos_embed
    # apply Transformer blocks
    for blk in self.decoder_blocks: x = blk(x)
    x = self.decoder_norm(x)
    # predictor projection
    x = self.decoder_pred(x)
    # remove cls token
    return x[:, 1:, :]

  def ci_forward(self:MaskedAutoencoderViT, x, n_splits=4):
    subsets, ids_restore = self.ci_forward_splitter(x, n_splits)

    preds, masks = [], []
    for z, mask, k in subsets:
      latent = self.ci_forward_encoder(z)
      pred = self.ci_forward_decoder(latent, ids_restore, k)  # [N, L, p*p*3]
      preds.append(pred)
      masks.append(mask)
    return preds, masks

  def cross_infer(self:MaskedAutoencoderViT, x:Tensor, n_split:int=8) -> Tensor:
    preds, masks = self.ci_forward(x, n_split)

    y = torch.zeros_like(preds[0])
    for p, m in zip(preds, masks):
      y = y + p * m.unsqueeze(-1)     # 1 is masked areas for prediction to fill
    return self.unpatchify(y)

  mae.ci_random_masking   = MethodType(ci_random_masking,   mae)
  mae.ci_forward_splitter = MethodType(ci_forward_splitter, mae)
  mae.ci_forward_encoder  = MethodType(ci_forward_encoder,  mae)
  mae.ci_forward_decoder  = MethodType(ci_forward_decoder,  mae)
  mae.ci_forward          = MethodType(ci_forward,          mae)
  mae.cross_infer         = MethodType(cross_infer,         mae)

  return mae


class MAE_dfn:

  def __init__(self):
    mae = mae_vit_large_patch16()
    mae.load_state_dict(torch.load(model_path, map_location=device)['model'])
    mae = mae.eval().to(device)
    self.model: MaskedAutoencoderViT = hijack_mae(mae)

  def __call__(self, x:Tensor, n_split:int=8) -> Tensor:
    return self.model.cross_infer(x, n_split)


if __name__ == '__main__':
  dfn = MAE_dfn()
  x = torch.rand([1, 3, 224, 224]).to(device)
  print('x.shape:', x.shape)
  y = dfn(x)
  print('y.shape:', y.shape)
  d = (x - y).abs().mean().item()
  print('err:', d)
