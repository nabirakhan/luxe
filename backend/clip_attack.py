"""CLIP Ensemble Attack — cosine similarity minimisation.

Loaded lazily. Ensemble of ViT-B/32 + ViT-L/14.
CLIP weights are fully frozen (eval + requires_grad=False).
Runs for ALL protection modes (nudify, modify, full).

attack(x_orig, x_pgd, eot) → x_clip_cloaked [1, 3, 512, 512]
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

import config
from utils import preprocess_for_clip

logger = logging.getLogger(__name__)


class CLIPEnsembleAttack:
    def __init__(self):
        self._b32 = None
        self._l14 = None
        self._lpips_fn = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load()

    def _load(self):
        import clip
        import lpips

        logger.info("Loading CLIP ViT-B/32...")
        b32, _ = clip.load("ViT-B/32", device=self._device)
        b32.eval()
        for p in b32.parameters():
            p.requires_grad_(False)
        self._b32 = b32

        logger.info("Loading CLIP ViT-L/14...")
        l14, _ = clip.load("ViT-L/14", device=self._device)
        l14.eval()
        for p in l14.parameters():
            p.requires_grad_(False)
        self._l14 = l14

        self._lpips_fn = lpips.LPIPS(net="alex").to(self._device)
        for p in self._lpips_fn.parameters():
            p.requires_grad_(False)

        logger.info("CLIPEnsembleAttack ready.")

    def attack(self, x_orig: Tensor, x_pgd: Tensor, eot) -> Tensor:
        """CLIP cosine-similarity minimisation on top of PGD delta.

        x_orig: [1, 3, 512, 512] original image, detached, [0, 1]
        x_pgd:  [1, 3, 512, 512] after PGD, detached, [0, 1]
        eot:    EOTTransforms instance

        Returns x_pgd + delta_clip clamped to [0, 1].
        """
        delta_pgd_linf = (x_pgd - x_orig).abs().max()
        eps_remaining = config.EPS_CLIP - delta_pgd_linf
        eps_remaining = eps_remaining.clamp(min=0)

        n_steps = 80 if eps_remaining < (6 / 255) else config.STEPS_CLIP

        # Reference embeddings computed ONCE — never changes during attack loop
        x_orig_pre = preprocess_for_clip(x_orig.detach())
        with torch.no_grad():
            b32_orig = self._b32.encode_image(x_orig_pre).float()  # [1, 512]
            l14_orig = self._l14.encode_image(x_orig_pre).float()  # [1, 768]

        delta_clip = torch.zeros_like(x_orig, requires_grad=True)

        for step in range(n_steps):
            x_eot = eot.apply_random(x_pgd + delta_clip)
            x_pre = preprocess_for_clip(x_eot)

            # Forward WITHOUT no_grad: grads flow to delta_clip through adversarial embeddings
            b32_adv = self._b32.encode_image(x_pre).float()
            l14_adv = self._l14.encode_image(x_pre).float()

            loss = (
                F.cosine_similarity(b32_orig, b32_adv, dim=-1).mean()
                + F.cosine_similarity(l14_orig, l14_adv, dim=-1).mean()
            )

            grad = torch.autograd.grad(loss, delta_clip)[0]

            # Descent: minimise cosine similarity
            delta_clip = (delta_clip - config.ALPHA * grad.sign()).detach()
            delta_clip = delta_clip.clamp(-eps_remaining, eps_remaining)
            delta_clip = delta_clip.requires_grad_(True)

            # LPIPS safety gate — rarely fires within ε budget
            if (step + 1) % config.CLIP_LPIPS_CHECK_EVERY == 0:
                with torch.no_grad():
                    score = self._lpips_fn(
                        x_orig * 2 - 1,
                        (x_pgd + delta_clip) * 2 - 1,
                    ).item()
                if score > config.LPIPS_THRESHOLD:
                    break

        return (x_pgd + delta_clip.detach()).clamp(0, 1)
