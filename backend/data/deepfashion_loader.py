"""DeepFashion dataset loader for adversarial training.

Images: 256×256 JPGs from Img/img.zip → /content/deepfashion/img/img/
Segmentation masks: color PNGs from img_highres_seg.zip → /content/deepfashion/seg/img_highres/
  Note: img_highres_seg.zip is at DeepFashion/img_highres_seg.zip on Drive,
  NOT inside Anno/segmentation/.

Attack binary mask: numpy color-match the following RGB values → 1, else → 0:
  (255,250,250) top      (250,235,215) skirt   (70,130,180) leggings
  (16,78,139) dress      (255,250,205) outer   (255,140,0)  pants
  (144,238,144) skin     (245,222,179) face

Preprocessing: direct resize to 512×512 (no centre-crop) to match protect.py
inference exactly — same distribution at train and inference time.

Train/val split from list_eval_partition.txt (evaluation_status == "train").
File has 3 columns: image_name, item_id, evaluation_status.
Paired views from list_item_inshop.txt.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Attack clothing RGB values
ATTACK_COLORS = [
    (255, 250, 250),   # top
    (250, 235, 215),   # skirt
    (70,  130, 180),   # leggings
    (16,  78,  139),   # dress
    (255, 250, 205),   # outer
    (255, 140,   0),   # pants
    (144, 238, 144),   # skin
    (245, 222, 179),   # face
]


def _build_attack_mask(seg_path: str) -> np.ndarray:
    """Return a binary H×W uint8 mask (1 = attack region)."""
    seg = np.array(Image.open(seg_path).convert("RGB"), dtype=np.uint8)
    mask = np.zeros(seg.shape[:2], dtype=np.uint8)
    for r, g, b in ATTACK_COLORS:
        match = (seg[:, :, 0] == r) & (seg[:, :, 1] == g) & (seg[:, :, 2] == b)
        mask[match] = 1
    return mask


class DeepFashionDataset(Dataset):
    """Returns (image_tensor, mask_tensor) pairs at 512×512."""

    def __init__(
        self,
        img_root: str,
        seg_root: str,
        partition_file: str,
        split: str = "train",
    ):
        self.img_root = Path(img_root)
        self.seg_root = Path(seg_root)
        self.split = split

        # Parse partition file.
        # File format (3 columns): image_name  item_id  evaluation_status
        # First line: total count. Second line: header. Data starts line 3.
        self.samples = []
        with open(partition_file) as f:
            lines = f.readlines()[2:]  # skip count line + header line
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            img_name = parts[0]   # e.g. img/WOMEN/Dresses/id_00000002/02_1_front.jpg
            status   = parts[2]   # train / val / test — parts[1] is item_id, NOT status
            if status == split:
                self.samples.append(img_name)

        print(f"DeepFashionDataset: {len(self.samples)} {split} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_name = self.samples[idx]

        # img_name is like "img/WOMEN/Dresses/id_00000002/02_1_front.jpg"
        # img_root is /content/deepfashion/img/img — strip leading "img/" from img_name
        rel = Path(img_name)
        parts = rel.parts
        # parts[0] is 'img', rest is the actual relative path
        rel_no_prefix = Path(*parts[1:]) if parts[0] == 'img' else rel

        img_path = self.img_root / rel_no_prefix
        img = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        # Segmentation mask — same relative path but .png extension
        seg_rel  = rel_no_prefix.with_suffix(".png")
        seg_path = self.seg_root / seg_rel
        if seg_path.exists():
            mask_np = _build_attack_mask(str(seg_path))
            mask_img = Image.fromarray(mask_np * 255).resize((512, 512), Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_img, dtype=np.float32) / 255.0).unsqueeze(0)
        else:
            mask_tensor = torch.ones(1, 512, 512, dtype=torch.float32)

        return img_tensor, mask_tensor


def get_loaders(img_root, seg_root, partition_file, batch_size=8):
    """Return (train_loader, val_loader, test_loader) with 80/10/10 split."""
    train_ds = DeepFashionDataset(img_root, seg_root, partition_file, split="train")
    val_ds   = DeepFashionDataset(img_root, seg_root, partition_file, split="val")
    test_ds  = DeepFashionDataset(img_root, seg_root, partition_file, split="test")

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader
