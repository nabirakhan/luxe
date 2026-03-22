"""SegFormer clothing mask generator.

Priority:
  1. checkpoints/segformer_lip.pth — fine-tuned 20-class LIP head
  2. nvidia/segformer-b2-finetuned-ade-512-512 — ADE20K fallback;
     maps only class 12 (person) → 1, all else → 0.

generate_mask(x) → [1, 1, 512, 512] binary float tensor.
LIP HIGH attack classes: {5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17}
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# LIP classes that correspond to clothing / skin regions to protect
LIP_HIGH_CLASSES = {5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17}

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "segformer_lip.pth"


class SegFormerMask:
    def __init__(self):
        self._model = None
        self._processor = None
        self._mode = None  # "lip" or "ade"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load()

    def _load(self):
        if CHECKPOINT_PATH.exists():
            self._load_lip()
        else:
            logger.warning(
                "segformer_lip.pth not found — falling back to ADE20K SegFormer. "
                "Mask quality will be lower (person class only)."
            )
            self._load_ade()

    def _load_lip(self):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        logger.info("Loading SegFormer with LIP checkpoint...")
        self._processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=20,
            ignore_mismatched_sizes=True,
        )
        state = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval().to(self._device)
        self._model = model
        self._mode = "lip"
        logger.info("SegFormer (LIP) loaded.")

    def _load_ade(self):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        logger.info("Loading SegFormer (ADE20K fallback)...")
        self._processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
        model.eval().to(self._device)
        self._model = model
        self._mode = "ade"
        logger.info("SegFormer (ADE20K) loaded.")

    @torch.no_grad()
    def generate_mask(self, x: Tensor) -> Tensor:
        """Return binary mask [1, 1, 512, 512] float on same device as x.

        x: [1, 3, 512, 512] float [0, 1]
        """
        # Convert to PIL for processor
        from PIL import Image
        import numpy as np

        img_np = (x.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pil_img = Image.fromarray(img_np)

        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        logits = self._model(**inputs).logits  # [1, num_classes, H', W']
        logits_up = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
        pred = logits_up.argmax(dim=1)  # [1, 512, 512]

        if self._mode == "lip":
            mask = torch.zeros_like(pred, dtype=torch.float32)
            for cls in LIP_HIGH_CLASSES:
                mask[pred == cls] = 1.0
        else:
            # ADE20K: class 12 = person
            mask = (pred == 12).float()

        return mask.unsqueeze(1).to(x.device)  # [1, 1, 512, 512]
