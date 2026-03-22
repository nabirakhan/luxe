"""Protector — main adversarial protection pipeline.

Lazy loads heavy models (InpaintLoss, ModificationLoss, CLIPEnsembleAttack,
CloakUNet) on first use. Only SegFormerMask loads at startup (~180MB).

No asyncio dependency — plain synchronous class.
Concurrency is handled in main.py at the HTTP layer.

Output resolution: always 512×512. Bilinear upsampling of an adversarial delta
destroys adversarial structure, so we never upsample back to original resolution.
"""

import io
import logging

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

import config
from eot import EOTTransforms
from segformer import SegFormerMask

logger = logging.getLogger(__name__)

UNET_CHECKPOINT = __import__("pathlib").Path(__file__).parent / "checkpoints" / "cloak_unet.pth"


class Protector:
    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading SegFormer (~180MB)...")
        self.segformer = SegFormerMask()

        self._eot = EOTTransforms(image_size=config.IMAGE_SIZE)

        # Lazily loaded
        self._inpaint_loss = None
        self._mod_loss = None
        self._clip_attack = None
        self._unet = None
        self._lpips_fn = None

    # ------------------------------------------------------------------
    # Lazy loaders — idempotent, safe to call 10k+ times
    # ------------------------------------------------------------------

    def _ensure_inpaint_loss(self):
        if self._inpaint_loss is None:
            logger.info("Loading InpaintLoss (~2GB)...")
            from pgd_inpaint import InpaintLoss
            self._inpaint_loss = InpaintLoss()

    def _ensure_mod_loss(self):
        if self._mod_loss is None:
            logger.info("Loading ModificationLoss...")
            from pgd_modification import ModificationLoss
            self._mod_loss = ModificationLoss()

    def _ensure_clip(self):
        if self._clip_attack is None:
            logger.info("Loading CLIP ensemble...")
            from clip_attack import CLIPEnsembleAttack
            self._clip_attack = CLIPEnsembleAttack()

    def _ensure_unet(self):
        if self._unet is None and UNET_CHECKPOINT.exists():
            logger.info("Loading CloakUNet...")
            from unet import CloakUNet
            unet = CloakUNet()
            unet.load_state_dict(torch.load(UNET_CHECKPOINT, map_location="cpu"))
            unet.eval().to(self._device)
            self._unet = unet

    def _ensure_lpips(self):
        if self._lpips_fn is None:
            import lpips
            self._lpips_fn = lpips.LPIPS(net="alex").to(self._device)
            for p in self._lpips_fn.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Preprocessing helper
    # ------------------------------------------------------------------

    def _load_image(self, image_bytes: bytes) -> torch.Tensor:
        """Load image bytes → [1, 3, 512, 512] float [0,1] on device.

        Direct resize to 512×512 (no centre-crop) — matches deepfashion_loader.py
        so U-Net inference sees the same distribution as training.
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS
        )
        arr = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)
        return x

    # ------------------------------------------------------------------
    # PGD-only path (also used by generate_unet_pairs.py)
    # ------------------------------------------------------------------

    def protect_pgd_only(
        self,
        image_bytes: bytes,
        mode: str,
        precomputed_mask: torch.Tensor | None = None,
    ):
        """Full PGD + CLIP adversarial protection.

        precomputed_mask: [1, 512, 512] float — passed from protect() to avoid
        a second SegFormer call. If None (standalone use), SegFormer runs here.

        Returns (x_orig_512, delta_final_512, mask_512):
            All [3, 512, 512] / [1, 512, 512] CPU tensors.
        """
        if mode not in ("nudify", "modify", "full"):
            raise ValueError(f"Unknown mode: {mode!r}")

        # Pre-load models with logging — idempotent, before the loop
        if mode in ("nudify", "full"):
            self._ensure_inpaint_loss()
        if mode in ("modify", "full"):
            self._ensure_mod_loss()
        self._ensure_clip()
        self._ensure_lpips()

        x = self._load_image(image_bytes)

        if precomputed_mask is not None:
            mask = precomputed_mask.to(self._device)  # [1, 512, 512]
        else:
            mask = self.segformer.generate_mask(x).squeeze(1)  # [1, 512, 512]

        x_orig_d = x.detach()
        delta = torch.zeros_like(x, requires_grad=True)

        # Joint PGD loop
        for _ in range(config.STEPS_PGD):
            x_eot = self._eot.apply_random(x_orig_d + delta)

            loss = torch.zeros(1, device=self._device, requires_grad=True)
            if mode in ("nudify", "full"):
                loss = loss + self._inpaint_loss.compute(x_orig_d, x_eot)
            if mode in ("modify", "full"):
                loss = loss + config.LAMBDA_MOD * self._mod_loss.compute(x_orig_d, x_eot)

            grad = torch.autograd.grad(loss, delta)[0]
            delta = (delta + config.ALPHA * grad.sign()).detach()
            delta = delta.clamp(-config.EPS_PGD, config.EPS_PGD)
            delta = delta * mask.unsqueeze(1)  # apply mask [1,1,512,512]
            delta = delta.requires_grad_(True)

        x_pgd = (x_orig_d + delta.detach()).clamp(0, 1)

        # CLIP attack — always, all modes
        x_cloaked = self._clip_attack.attack(x_orig_d, x_pgd, self._eot)

        delta_final = (x_cloaked - x_orig_d).detach()

        # LPIPS auto-scale safety gate
        with torch.no_grad():
            score = self._lpips_fn(x_orig_d * 2 - 1, x_cloaked * 2 - 1).item()
        if score > config.LPIPS_THRESHOLD:
            logger.warning(f"LPIPS {score:.4f} > threshold — scaling delta down.")
            delta_final = delta_final * (config.LPIPS_THRESHOLD / score)

        return (
            x_orig_d.squeeze(0).cpu(),   # [3, 512, 512]
            delta_final.squeeze(0).cpu(), # [3, 512, 512]
            mask.squeeze(0).cpu(),        # [1, 512, 512]
        )

    # ------------------------------------------------------------------
    # Main protection entry point
    # ------------------------------------------------------------------

    def protect(
        self,
        image_bytes: bytes,
        mode: str,
        texture: bool = False,
    ):
        """Protect an image — returns (png_bytes, checkpoint_status).

        Fast path: CloakUNet (<10s). Fallback: full PGD (~90s/image).
        Output is always 512×512 PNG.
        """
        if mode not in ("nudify", "modify", "full"):
            raise ValueError(f"Unknown mode: {mode!r}")

        x = self._load_image(image_bytes)

        # SegFormer called ONCE at the top
        # squeeze(1): [1,1,512,512] → [1,512,512]  (collapse channel dim, not batch)
        mask = self.segformer.generate_mask(x).squeeze(1)  # [1, 512, 512]

        self._ensure_unet()

        if self._unet is not None:
            # Fast path
            with torch.no_grad():
                unet_out = self._unet(x)  # [1, 3, 512, 512]
            delta_512 = unet_out.squeeze(0) * mask.squeeze(0)  # [3, 512, 512]
            checkpoint_status = "UNET_OK"
        else:
            logger.warning(
                "WARNING: cloak_unet.pth missing — est. load 3–5 min, inference ~90s/image"
            )
            x_orig_512, delta_512, _ = self.protect_pgd_only(
                image_bytes, mode, precomputed_mask=mask
            )
            delta_512 = delta_512.to(self._device)
            checkpoint_status = "UNET_MISSING"

        result = (x.squeeze(0) + delta_512).clamp(0, 1)  # [3, 512, 512]

        if texture:
            from texture import TexturePerturbation
            tp = TexturePerturbation()
            result = tp.apply(result, mask.squeeze(0))

        # Encode to PNG
        result_np = (result.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_out = Image.fromarray(result_np)
        buf = io.BytesIO()
        pil_out.save(buf, format="PNG")
        return buf.getvalue(), checkpoint_status

    def checkpoint_status(self) -> dict:
        return {
            "segformer_lip": (
                __import__("pathlib").Path(__file__).parent / "checkpoints" / "segformer_lip.pth"
            ).exists(),
            "cloak_unet": UNET_CHECKPOINT.exists(),
            "sd_inpaint_vae": (
                __import__("pathlib").Path(__file__).parent / "checkpoints" / "sd_inpaint_vae.pth"
            ).exists(),
            "ipp_vae": (
                __import__("pathlib").Path(__file__).parent / "checkpoints" / "ipp_vae.pth"
            ).exists(),
            "ip_adapter": (
                __import__("pathlib").Path(__file__).parent / "checkpoints" / "ip_adapter.pth"
            ).exists(),
        }
