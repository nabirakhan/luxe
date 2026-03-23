"""Look Into Person (LIP) dataset loader for SegFormer training.

<<<<<<< HEAD
Actual structure after unzipping:
  /content/lip/images/train_images/*.jpg
  /content/lip/images/val_images/*.jpg
  /content/lip/annotations/train_segmentations/*.png
  /content/lip/annotations/val_segmentations/*.png
=======
Images: /content/lip/images/TrainVal_images/
Annotations (label maps 0–19): /content/lip/annotations/TrainVal_parsing_annotations/
>>>>>>> a5c21b168267f2ba1cd12a10ad4ffb36071921c6

HIGH-priority attack classes → 1, all others → 0:
  {5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17}

LIP structure after unzip:
  /content/lip/images/TrainVal_images/{train,val}/images/*.jpg
  /content/lip/annotations/TrainVal_parsing_annotations/{train,val}/annotations/*.png

train_id.txt and val_id.txt list the stems for each split.
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

HIGH_ATTACK_CLASSES = {5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17}

# Map split name → actual folder name after unzip
IMG_SPLIT_DIRS = {
    "train": "train_images",
    "val":   "val_images",
}
ANN_SPLIT_DIRS = {
    "train": "train_segmentations",
    "val":   "val_segmentations",
}


def _find_split_dirs(img_dir: Path, ann_dir: Path, split: str):
    """
    Find the actual image and annotation directories for a split.
    Handles varying unzip structures robustly.
    """
    # Common patterns after unzip
    img_candidates = [
        img_dir / split / "images",
        img_dir / "TrainVal_images" / split / "images",
        img_dir / split,
        img_dir,
    ]
    ann_candidates = [
        ann_dir / split / "annotations",
        ann_dir / "TrainVal_parsing_annotations" / split / "annotations",
        ann_dir / split,
        ann_dir,
    ]

    img_found = next((p for p in img_candidates if p.exists()), None)
    ann_found = next((p for p in ann_candidates if p.exists()), None)

    if img_found is None:
        raise FileNotFoundError(
            f"Could not find LIP {split} images under {img_dir}. "
            f"Tried: {img_candidates}"
        )
    if ann_found is None:
        raise FileNotFoundError(
            f"Could not find LIP {split} annotations under {ann_dir}. "
            f"Tried: {ann_candidates}"
        )

    return img_found, ann_found


class LIPDataset(Dataset):
    """Returns (image_tensor [3,512,512], label_tensor [1,512,512] long) pairs.

<<<<<<< HEAD
    label_tensor: multi-class 0-19 for SegFormer cross-entropy training.
    """

    def __init__(self, img_dir: str, ann_dir: str, split: str = "train"):
        self.split    = split
        img_root      = Path(img_dir)
        ann_root      = Path(ann_dir)

        img_split_dir = IMG_SPLIT_DIRS[split]
        ann_split_dir = ANN_SPLIT_DIRS[split]

        self.img_search = img_root / img_split_dir
        self.ann_search = ann_root / ann_split_dir

        if not self.img_search.exists():
            raise FileNotFoundError(
                f"LIP image dir not found: {self.img_search}\n"
                f"Contents of {img_root}: {list(img_root.iterdir())}"
            )
        if not self.ann_search.exists():
            raise FileNotFoundError(
                f"LIP annotation dir not found: {self.ann_search}\n"
                f"Contents of {ann_root}: {list(ann_root.iterdir())}"
            )

        # Build stem list from jpg files
        self.stems = sorted([p.stem for p in self.img_search.glob("*.jpg")])
        print(f"LIPDataset [{split}]: {len(self.stems)} images in {self.img_search}")
=======
    label_tensor: multi-class long 0-19 for SegFormer cross-entropy training.
    """

    def __init__(self, img_dir: str, ann_dir: str, split: str = "train"):
        self.split = split
        img_root = Path(img_dir)
        ann_root = Path(ann_dir)

        img_search, ann_search = _find_split_dirs(img_root, ann_root, split)
        self.img_search = img_search
        self.ann_search = ann_search

        # Collect all jpg stems
        self.stems = sorted([p.stem for p in img_search.rglob("*.jpg")])
        print(f"LIPDataset [{split}]: {len(self.stems)} images in {img_search}")
>>>>>>> a5c21b168267f2ba1cd12a10ad4ffb36071921c6

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]

        # Image
<<<<<<< HEAD
        img_path = self.img_search / f"{stem}.jpg"
=======
        img_path = next(self.img_search.rglob(f"{stem}.jpg"), None)
        if img_path is None:
            raise FileNotFoundError(f"LIP image not found: {stem}.jpg under {self.img_search}")
>>>>>>> a5c21b168267f2ba1cd12a10ad4ffb36071921c6
        img = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        img_tensor = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)  # [3, 512, 512]

        # Label map (PNG, integer values 0-19)
<<<<<<< HEAD
        ann_path = self.ann_search / f"{stem}.png"
        if ann_path.exists():
            label_np = np.array(
                Image.open(ann_path).resize((512, 512), Image.NEAREST),
                dtype=np.int64
            )
            label_tensor = torch.from_numpy(label_np).unsqueeze(0)  # [1, 512, 512]
        else:
            label_tensor = torch.zeros(1, 512, 512, dtype=torch.long)
=======
        ann_path = next(self.ann_search.rglob(f"{stem}.png"), None)
        if ann_path is None:
            label_tensor = torch.zeros(1, 512, 512, dtype=torch.long)
        else:
            label_np = np.array(
                Image.open(ann_path).resize((512, 512), Image.NEAREST),
                dtype=np.int64
            )
            label_tensor = torch.from_numpy(label_np).unsqueeze(0)  # [1, 512, 512]
>>>>>>> a5c21b168267f2ba1cd12a10ad4ffb36071921c6

        return img_tensor, label_tensor


class LIPBinaryDataset(LIPDataset):
<<<<<<< HEAD
    """Same as LIPDataset but returns binary float mask instead of full label map."""
=======
    """Same as LIPDataset but returns a binary float mask instead of full label map."""
>>>>>>> a5c21b168267f2ba1cd12a10ad4ffb36071921c6

    def __getitem__(self, idx: int):
        img_tensor, label_tensor = super().__getitem__(idx)
        binary = torch.zeros_like(label_tensor, dtype=torch.float32)
        for cls in HIGH_ATTACK_CLASSES:
            binary[label_tensor == cls] = 1.0
        return img_tensor, binary


def get_lip_loaders(img_dir, ann_dir, batch_size=8, binary=False, **kwargs):
    """Return (train_loader, val_loader).
<<<<<<< HEAD

    kwargs passed through to DataLoader — supports num_workers, pin_memory,
=======
    
    kwargs passed to DataLoader — supports num_workers, pin_memory,
>>>>>>> a5c21b168267f2ba1cd12a10ad4ffb36071921c6
    persistent_workers, prefetch_factor.
    """
    cls = LIPBinaryDataset if binary else LIPDataset
    train_ds = cls(img_dir, ann_dir, split="train")
    val_ds   = cls(img_dir, ann_dir, split="val")

    loader_kwargs = dict(num_workers=2, pin_memory=True)
    loader_kwargs.update(kwargs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)
<<<<<<< HEAD
    return train_loader, val_loader
=======
    return train_loader, val_loader
>>>>>>> a5c21b168267f2ba1cd12a10ad4ffb36071921c6
