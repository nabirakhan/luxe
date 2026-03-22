"""Evaluation metrics.

All functions accept PIL Images or [H,W,C] / [C,H,W] numpy/tensor inputs in [0,1].
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils import preprocess_for_clip


def _to_numpy(x) -> np.ndarray:
    """Convert PIL Image or tensor to HWC uint8 numpy array."""
    if isinstance(x, Image.Image):
        return np.array(x.convert("RGB"), dtype=np.float32) / 255.0
    if isinstance(x, torch.Tensor):
        t = x.float()
        if t.dim() == 4:
            t = t.squeeze(0)
        if t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        return t.cpu().numpy()
    return np.array(x, dtype=np.float32)


def _to_bchw(x) -> torch.Tensor:
    """Convert input to [1, C, H, W] float tensor in [0, 1]."""
    if isinstance(x, Image.Image):
        arr = np.array(x.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if isinstance(x, torch.Tensor):
        t = x.float()
        if t.dim() == 3:
            t = t.unsqueeze(0)
        return t
    arr = np.array(x, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


# ------------------------------------------------------------------
# LPIPS
# ------------------------------------------------------------------

_lpips_model = None


def _get_lpips():
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex")
        _lpips_model.eval()
    return _lpips_model


def lpips_score(a, b) -> float:
    """AlexNet LPIPS distance (lower = more similar). Inputs [0, 1]."""
    model = _get_lpips()
    ta = _to_bchw(a) * 2 - 1
    tb = _to_bchw(b) * 2 - 1
    with torch.no_grad():
        return model(ta, tb).item()


# ------------------------------------------------------------------
# SSIM
# ------------------------------------------------------------------

def ssim_score(a, b) -> float:
    """Structural Similarity Index (higher = more similar)."""
    from skimage.metrics import structural_similarity
    na = _to_numpy(a)
    nb = _to_numpy(b)
    return structural_similarity(na, nb, data_range=1.0, channel_axis=-1)


# ------------------------------------------------------------------
# PSNR
# ------------------------------------------------------------------

def psnr_score(a, b) -> float:
    """Peak Signal-to-Noise Ratio in dB (higher = more similar)."""
    from skimage.metrics import peak_signal_noise_ratio
    na = _to_numpy(a)
    nb = _to_numpy(b)
    return peak_signal_noise_ratio(na, nb, data_range=1.0)


# ------------------------------------------------------------------
# CLIP cosine similarity
# ------------------------------------------------------------------

_clip_b32 = None


def _get_clip():
    global _clip_b32
    if _clip_b32 is None:
        import clip
        model, _ = clip.load("ViT-B/32", device="cpu")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _clip_b32 = model
    return _clip_b32


def clip_cosine_sim(a, b) -> float:
    """CLIP ViT-B/32 cosine similarity between two images (higher = more similar).

    Uses preprocess_for_clip from utils (avoids circular import with clip_attack).
    """
    model = _get_clip()
    ta = _to_bchw(a)
    tb = _to_bchw(b)

    ta_pre = preprocess_for_clip(ta)
    tb_pre = preprocess_for_clip(tb)

    with torch.no_grad():
        ea = model.encode_image(ta_pre).float()
        eb = model.encode_image(tb_pre).float()

    return F.cosine_similarity(ea, eb, dim=-1).mean().item()
