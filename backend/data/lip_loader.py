"""Look Into Person (LIP) dataset loader for SegFormer training.

Images: /content/lip/images/
Annotations (label maps 0–19): /content/lip/annotations/

HIGH-priority attack classes → 1, all others → 0:
  {5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17}
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

HIGH_ATTACK_CLASSES = {5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17}


class LIPDataset(Dataset):
    """Returns (image_tensor [3,512,512], label_tensor [1,512,512]) pairs.

    label_tensor: multi-class 0-19 for SegFormer training head.
    binary_mask: HIGH_ATTACK_CLASSES → 1, rest → 0.
    """

    def __init__(self, img_dir: str, ann_dir: str, split: str = "train"):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.split = split

        # Collect image stems
        search_dir = self.img_dir / split if (self.img_dir / split).exists() else self.img_dir
        self.stems = sorted([p.stem for p in search_dir.glob("*.jpg")])

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]

        # Image — direct resize to 512×512
        img_path = next(self.img_dir.rglob(f"{stem}.jpg"), None)
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {stem}.jpg")
        img = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        # Label map (PNG, values 0-19)
        ann_path = next(self.ann_dir.rglob(f"{stem}.png"), None)
        if ann_path is None:
            label_tensor = torch.zeros(1, 512, 512, dtype=torch.long)
        else:
            label_np = np.array(Image.open(ann_path).resize((512, 512), Image.NEAREST), dtype=np.int64)
            label_tensor = torch.from_numpy(label_np).unsqueeze(0)

        return img_tensor, label_tensor


class LIPBinaryDataset(LIPDataset):
    """Same as LIPDataset but returns a binary mask instead of full label map."""

    def __getitem__(self, idx: int):
        img_tensor, label_tensor = super().__getitem__(idx)
        binary = torch.zeros_like(label_tensor)
        for cls in HIGH_ATTACK_CLASSES:
            binary[label_tensor == cls] = 1
        return img_tensor, binary.float()


def get_lip_loaders(img_dir, ann_dir, batch_size=8, binary=False):
    """Return (train_loader, val_loader)."""
    cls = LIPBinaryDataset if binary else LIPDataset
    train_ds = cls(img_dir, ann_dir, split="train")
    val_ds   = cls(img_dir, ann_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader
