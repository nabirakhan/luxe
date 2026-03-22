"""TexturePerturbation — tiled sine-wave fabric micro-texture.

Applies to clothing regions only (via mask). Never used during eval.

CLI calibration mode:
    python texture.py --calibrate --img_dir /path/to/images --seg_dir /path/to/segs

Prints LPIPS per amplitude on 10 representative images. Confirm
TEXTURE_AMPLITUDE < 0.10 threshold before demo.
"""

import argparse

import numpy as np
import torch
from torch import Tensor

import config


class TexturePerturbation:
    """Tiled sine-wave fabric micro-texture."""

    def __init__(self, amplitude: float | None = None):
        self.amplitude = amplitude if amplitude is not None else config.TEXTURE_AMPLITUDE

    def apply(self, image: Tensor, mask: Tensor) -> Tensor:
        """Apply texture to clothing regions.

        image: [3, H, W] float [0, 1]
        mask:  [1, H, W] or [H, W] binary float
        Returns [3, H, W] float [0, 1]
        """
        _, H, W = image.shape
        device = image.device

        # Tiled sine-wave: two perpendicular frequencies
        xs = torch.linspace(0, 4 * np.pi, W, device=device)
        ys = torch.linspace(0, 4 * np.pi, H, device=device)
        grid_x, grid_y = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

        texture = self.amplitude * (
            0.5 * torch.sin(grid_x) + 0.5 * torch.sin(grid_y)
        )  # [H, W], values in [-amplitude, +amplitude]

        # Expand to RGB and apply only to masked regions
        texture_rgb = texture.unsqueeze(0).expand(3, -1, -1)  # [3, H, W]
        if mask.dim() == 3:
            mask_2d = mask.squeeze(0)  # [H, W]
        else:
            mask_2d = mask  # [H, W]

        mask_rgb = mask_2d.unsqueeze(0).expand(3, -1, -1)  # [3, H, W]
        return (image + texture_rgb * mask_rgb).clamp(0, 1)


# ------------------------------------------------------------------
# CLI calibration
# ------------------------------------------------------------------

def _calibrate(img_dir: str, seg_dir: str, n: int = 10):
    """Run on n images, print LPIPS per amplitude. Confirm < 0.10 threshold."""
    import os
    from pathlib import Path
    from PIL import Image as PILImage
    from data.deepfashion_loader import _build_attack_mask

    import lpips
    lpips_fn = lpips.LPIPS(net="alex")
    lpips_fn.eval()

    img_paths = sorted(Path(img_dir).rglob("*.jpg"))[:n]
    if not img_paths:
        print("No .jpg images found.")
        return

    amplitudes = [0.005, 0.010, 0.015, 0.020, 0.025]

    for amp in amplitudes:
        tp = TexturePerturbation(amplitude=amp)
        scores = []
        for img_path in img_paths:
            # Load image
            img = PILImage.open(img_path).convert("RGB").resize((512, 512), PILImage.LANCZOS)
            img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

            # Try to find matching seg mask
            stem = img_path.stem
            seg_path = next(Path(seg_dir).rglob(f"{stem}.png"), None)
            if seg_path:
                mask_np = _build_attack_mask(str(seg_path))
                mask_img = PILImage.fromarray(mask_np * 255).resize((512, 512), PILImage.NEAREST)
                mask_t = torch.from_numpy(np.array(mask_img, dtype=np.float32) / 255.0).unsqueeze(0)
            else:
                mask_t = torch.ones(1, 512, 512)

            out_t = tp.apply(img_t, mask_t)

            a = (img_t.unsqueeze(0) * 2 - 1)
            b = (out_t.unsqueeze(0) * 2 - 1)
            with torch.no_grad():
                score = lpips_fn(a, b).item()
            scores.append(score)

        mean_lpips = np.mean(scores)
        ok = "OK" if mean_lpips < config.LPIPS_THRESHOLD_TEXTURE else "FAIL"
        print(f"amplitude={amp:.3f}  LPIPS={mean_lpips:.4f}  [{ok}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--img_dir", default="/content/deepfashion/img")
    parser.add_argument("--seg_dir", default="/content/deepfashion/seg")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    if args.calibrate:
        _calibrate(args.img_dir, args.seg_dir, n=args.n)
    else:
        print("Run with --calibrate to measure LPIPS per amplitude.")
