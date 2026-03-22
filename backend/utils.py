import torch
import torch.nn.functional as F
from torch import Tensor

from config import CLIP_MEAN, CLIP_STD, CLIP_SIZE


def preprocess_for_clip(x: Tensor) -> Tensor:
    """Resize and normalise a [B,3,H,W] float [0,1] tensor for CLIP.

    Used by both clip_attack.py and metrics.py. Lives here to avoid the
    circular import that would arise if metrics imported from clip_attack.
    """
    x = F.interpolate(x, size=(CLIP_SIZE, CLIP_SIZE), mode="bicubic", align_corners=False)
    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(CLIP_STD,  device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std
