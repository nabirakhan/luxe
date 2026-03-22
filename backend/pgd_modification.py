"""ModificationLoss — IPP + IP-Adapter disruption for outfit-swap attacks.

Loaded lazily. Combines:
  - IPP VAE latent distance
  - IP-Adapter CLIP image projection distance

Loss direction: MAXIMIZE (gradient ascent in the PGD loop).
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from config import LAMBDA_IPA

logger = logging.getLogger(__name__)

IPP_CHECKPOINT = Path(__file__).parent / "checkpoints" / "ipp_vae.pth"
IPA_CHECKPOINT = Path(__file__).parent / "checkpoints" / "ip_adapter.pth"


class ModificationLoss:
    def __init__(self):
        self._ipp_vae = None
        self._ipa_proj = None
        self._clip_model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load()

    def _load(self):
        self._load_ipp_vae()
        self._load_ipa()

    def _load_ipp_vae(self):
        from diffusers import AutoencoderKL

        logger.info("Loading IPP VAE...")
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")

        if IPP_CHECKPOINT.exists():
            state = torch.load(IPP_CHECKPOINT, map_location="cpu")
            vae.load_state_dict(state, strict=False)
            logger.info("Loaded fine-tuned IPP VAE weights.")
        else:
            logger.warning("ipp_vae.pth not found — using vanilla SD VAE for ModificationLoss.")

        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        self._ipp_vae = vae.to(self._device)

    def _load_ipa(self):
        import clip

        logger.info("Loading IP-Adapter CLIP + image projection...")
        clip_model, _ = clip.load("ViT-L/14", device=self._device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)
        self._clip_model = clip_model

        # Image projection head (linear: 1024 → 768 by default in IP-Adapter)
        if IPA_CHECKPOINT.exists():
            state = torch.load(IPA_CHECKPOINT, map_location="cpu")
            # Extract image_proj_model weights if the checkpoint is the full IP-Adapter state dict
            proj_weights = {k.replace("image_proj_model.", ""): v
                           for k, v in state.items() if "image_proj_model" in k}
            if proj_weights:
                in_dim  = next(iter(proj_weights.values())).shape[-1]
                out_dim = list(proj_weights.values())[-1].shape[0]
                proj = torch.nn.Linear(in_dim, out_dim, bias=False)
                proj.load_state_dict(proj_weights, strict=False)
                logger.info("IP-Adapter image_proj_model loaded.")
            else:
                proj = torch.nn.Identity()
                logger.warning("ip_adapter.pth found but no image_proj_model keys — using Identity.")
        else:
            logger.warning("ip_adapter.pth not found — using Identity projection for ModificationLoss.")
            proj = torch.nn.Identity()

        proj.eval()
        for p in proj.parameters():
            p.requires_grad_(False)
        self._ipa_proj = proj.to(self._device)

    def _encode_ipp_vae(self, x: Tensor) -> Tensor:
        scaled = x * 2 - 1
        return self._ipp_vae.encode(scaled).latent_dist.mean

    def _encode_ipa(self, x: Tensor) -> Tensor:
        import torch
        # CLIP ViT-L/14 expects 224×224 normalised inputs — use preprocess_for_clip
        from utils import preprocess_for_clip
        x_pre = preprocess_for_clip(x)
        feat = self._clip_model.encode_image(x_pre).float()
        return self._ipa_proj(feat)

    def compute(self, x_orig_d: Tensor, x_adv: Tensor) -> Tensor:
        """Combined IPP-VAE + IP-Adapter loss (to be maximised).

        x_orig_d: detached original [1, 3, 512, 512] in [0, 1]
        x_adv:    adversarial (requires_grad) [1, 3, 512, 512] in [0, 1]
        """
        with torch.no_grad():
            vae_orig = self._encode_ipp_vae(x_orig_d)
            ipa_orig = self._encode_ipa(x_orig_d)

        vae_adv = self._encode_ipp_vae(x_adv)
        ipa_adv = self._encode_ipa(x_adv)

        loss_vae = F.mse_loss(vae_orig, vae_adv)
        loss_ipa = F.mse_loss(ipa_orig, ipa_adv)

        return LAMBDA_IPA * loss_vae + (1 - LAMBDA_IPA) * loss_ipa
