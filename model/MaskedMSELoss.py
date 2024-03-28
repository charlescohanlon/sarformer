import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# likely loss function used in 4M (from MultiMAE)
# From https://github.com/EPFL-VILAB/MultiMAE/blob/main/multimae/criterion.py


class MaskedMSELoss(nn.Module):
    """L1 loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param norm_pix: Normalized pixel loss
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.norm_pix = norm_pix

    def patchify(self, imgs, nh, nw):
        p = self.scale_factor
        x = rearrange(
            imgs, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p
        )
        return x

    def unpatchify(self, x, nh, nw):
        p = self.scale_factor
        imgs = rearrange(
            x, "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)", nh=nh, nw=nw, p1=p, p2=p
        )
        return imgs

    def forward(self, input, target, mask=None):

        H, W = input.shape[-2:]
        nh, nw = H // self.scale_factor, W // self.scale_factor

        if self.norm_pix:
            target = self.patchify(target, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nh, nw)

        loss = F.mse_loss(input, target, reduction="none")

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(
                mask.unsqueeze(1).float(), size=(H, W), mode="nearest"
            ).squeeze(1)
            loss = loss.mean(dim=1)  # B, C, H, W -> B, H, W
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(
                dim=1
            )
            loss = loss.nanmean()  # Account for zero masks
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training

        return loss
