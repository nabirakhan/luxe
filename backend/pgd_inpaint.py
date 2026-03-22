"""InpaintLoss — VAE latent-space disruption for nudification attacks.

Loaded lazily. Uses the fine-tuned SD v1.5 VAE (falls back to vanilla).
VAE is frozen and kept in eval mode.

Loss direction: MAXIMIZE MSE between original and adversarial latents
(gradient ascent in the PGD loop).
"""

import logging
from pathlib import Path

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "sd_inpaint_vae.pth"


class InpaintLoss:
    def __init__(self):
        self._vae = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load()

    def _load(self):
        from diffusers import AutoencoderKL

        if CHECKPOINT_PATH.exists():
            logger.info("Loading fine-tuned SD inpaint VAE...")
            vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
            state = torch.load(CHECKPOINT_PATH, map_location="cpu")
            vae.load_state_dict(state, strict=False)
        else:
            logger.warning("sd_inpaint_vae.pth not found — using vanilla SD v1.5 VAE.")
            vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")

        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        self._vae = vae.to(self._device)
        logger.info("InpaintLoss VAE ready.")

    def compute(self, x_orig_d: Tensor, x_adv: Tensor) -> Tensor:
        """MSE between original and adversarial VAE latents.

        x_orig_d: detached original image [1, 3, 512, 512] in [0, 1]
        x_adv:    adversarial image (requires_grad) [1, 3, 512, 512] in [0, 1]

        Returns scalar loss (to be maximised via gradient ascent).
        """
        # Rescale from [0,1] to [-1,1] as expected by SD VAE
        orig_scaled = x_orig_d * 2 - 1
        adv_scaled  = x_adv   * 2 - 1

        with torch.no_grad():
            lat_orig = self._vae.encode(orig_scaled).latent_dist.mean

        lat_adv = self._vae.encode(adv_scaled).latent_dist.mean

        return torch.nn.functional.mse_loss(lat_orig, lat_adv)
