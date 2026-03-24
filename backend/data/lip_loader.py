"""Look Into Person (LIP) dataset loader for SegFormer training.

Zip extracts to:
  /content/lip/images/train_images/*.jpg          (30,462 images)
  /content/lip/images/val_images/*.jpg            (10,000 images)
  /content/lip/annotations/train_segmentations/*.png  (30,462 masks)
  /content/lip/annotations/val_segmentations/*.png    (10,000 masks)

Train/val split from train_id.txt and val_id.txt.
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

HIGH_ATTACK_CLASSES = {5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17}

# Works on both Colab and Kaggle
def _find_id_txt(filename):
    candidates = [
        f"/content/lip/{filename}",
        f"/content/drive/MyDrive/dlp_project/DLP Project Datasets/Look Into Person/TrainVal_images/{filename}",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"{filename} not found. Tried: {candidates}")


def _load_stems(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


class LIPDataset(Dataset):
    def __init__(self, img_dir: str, ann_dir: str, split: str = "train"):
        if split == "train":
            self.img_dir = Path(img_dir) / "train_images"
            self.ann_dir = Path(ann_dir) / "train_segmentations"
            id_txt = _find_id_txt("train_id.txt")
        else:
            self.img_dir = Path(img_dir) / "val_images"
            self.ann_dir = Path(ann_dir) / "val_segmentations"
            id_txt = _find_id_txt("val_id.txt")

        all_stems = _load_stems(id_txt)
        available = {p.stem for p in self.img_dir.glob("*.jpg")}
        self.stems = [s for s in all_stems if s in available]
        print(f"[LIPDataset] split={split}  found={len(self.stems)}/{len(all_stems)}")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]

        img = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB").resize((512, 512), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        ann_path = self.ann_dir / f"{stem}.png"
        if ann_path.exists():
            label_np = np.array(Image.open(ann_path).resize((512, 512), Image.NEAREST), dtype=np.int64)
            label_tensor = torch.from_numpy(label_np).unsqueeze(0)
        else:
            label_tensor = torch.zeros(1, 512, 512, dtype=torch.long)

        return img_tensor, label_tensor


class LIPBinaryDataset(LIPDataset):
    def __getitem__(self, idx):
        img_tensor, label_tensor = super().__getitem__(idx)
        binary = torch.zeros_like(label_tensor, dtype=torch.float32)
        for cls in HIGH_ATTACK_CLASSES:
            binary[label_tensor == cls] = 1.0
        return img_tensor, binary


def get_lip_loaders(img_dir, ann_dir, batch_size=8, binary=False):
    cls = LIPBinaryDataset if binary else LIPDataset
    train_ds = cls(img_dir, ann_dir, split="train")
    val_ds   = cls(img_dir, ann_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader
