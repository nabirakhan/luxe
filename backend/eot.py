"""Expectation over Transformations (EOT) — differentiable image augmentations.

All transforms operate on [B, C, H, W] float tensors in [0, 1].
Gradients flow through every operation; no detach() inside transforms.
"""

import random

import torch
import torch.nn.functional as F
from torch import Tensor


class EOTTransforms:
    def __init__(self, image_size: int = 512):
        self.image_size = image_size

    # ------------------------------------------------------------------
    # Individual transforms
    # ------------------------------------------------------------------

    def jpeg_simulate(self, x: Tensor) -> Tensor:
        """Simulate JPEG compression via blur + noise."""
        q = random.uniform(50, 95)
        sigma = (100 - q) / 100 * 2
        noise_std = (100 - q) / 100 * 0.03

        # Gaussian blur with kernel size 5
        k = 5
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = gauss_1d[:, None] * gauss_1d[None, :]
        kernel = kernel.view(1, 1, k, k).expand(x.shape[1], 1, k, k)

        x_blur = F.conv2d(x, kernel, padding=k // 2, groups=x.shape[1])
        noise = torch.randn_like(x_blur) * noise_std
        return (x_blur + noise).clamp(0, 1)

    def random_resize(self, x: Tensor) -> Tensor:
        """Randomly downscale then upscale back to original size."""
        scale = random.uniform(0.75, 1.0)
        h = max(1, int(self.image_size * scale))
        w = max(1, int(self.image_size * scale))
        x_small = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False)
        return F.interpolate(x_small, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False).clamp(0, 1)

    def random_crop_pad(self, x: Tensor) -> Tensor:
        """Random crop then pad back — gradient flows through slice+pad."""
        crop_frac = random.uniform(0.75, 1.0)
        ch = max(1, int(self.image_size * crop_frac))
        cw = max(1, int(self.image_size * crop_frac))

        # Sample offsets outside the computation graph
        top  = random.randint(0, self.image_size - ch)
        left = random.randint(0, self.image_size - cw)

        # Differentiable slice
        cropped = x[:, :, top: top + ch, left: left + cw]

        # Pad back to original size (differentiable)
        pad_bottom = self.image_size - ch
        pad_right  = self.image_size - cw
        # F.pad order: (left, right, top, bottom)
        return F.pad(cropped, (left, pad_right, top, pad_bottom), mode="constant", value=0)

    def screenshot_simulate(self, x: Tensor) -> Tensor:
        """Simulate screen capture: gamma encode → blur → noise → gamma decode."""
        gamma = 2.2
        x_enc = x.clamp(1e-6, 1.0) ** (1.0 / gamma)

        # Light Gaussian blur
        k = 3
        sigma = 0.5
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = (gauss_1d[:, None] * gauss_1d[None, :]).view(1, 1, k, k).expand(x.shape[1], 1, k, k)
        x_blur = F.conv2d(x_enc, kernel, padding=k // 2, groups=x.shape[1])

        noise = torch.randn_like(x_blur) * 0.02
        x_noisy = (x_blur + noise).clamp(1e-6, 1.0)
        return (x_noisy ** gamma).clamp(0, 1)

    def color_jitter(self, x: Tensor) -> Tensor:
        """Per-channel contrast + brightness jitter."""
        factor = random.uniform(0.8, 1.2)
        bright = random.uniform(0.8, 1.2)

        mean = x.mean(dim=[-2, -1], keepdim=True)
        x_jit = mean + factor * (x - mean)
        x_jit = (x_jit * bright).clamp(0, 1)
        return x_jit

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def apply_random(self, x: Tensor) -> Tensor:
        """Apply 1–3 randomly chosen transforms in sequence."""
        transforms = [
            self.jpeg_simulate,
            self.random_resize,
            self.random_crop_pad,
            self.screenshot_simulate,
            self.color_jitter,
        ]
        chosen = random.sample(transforms, k=random.randint(1, 3))
        for t in chosen:
            x = t(x)
        return x
