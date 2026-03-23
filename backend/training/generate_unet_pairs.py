"""Generate (image, pgd_delta) pairs for U-Net distillation training.

Mode is always "full" — trains the U-Net on combined nudify+modify protection.
Runtime: ~5-10s/image on T4 -> 14-28 hours for 10k images. Plan 3+ sessions.

Features:
- Resume-safe: skips images where data/unet_pairs/{stem}.pt already exists.
- Per-image error handling: logs failures and continues (never crashes the run).
- Drive checkpoint after every 100 pairs.
- Strips batch dim before saving: DataLoader expects [3,512,512] not [1,3,512,512].

Run on Colab T4. Unzip cell at top of session (see train_segformer.py).
"""

from google.colab import drive
import os
drive.mount('/content/drive')

import sys


def find_drive_base():
    candidates = [
        '/content/drive/MyDrive/Datasets/DLP Project Datasets',
        '/content/drive/MyDrive/DLP Project Datasets',
        '/content/drive/MyDrive/DLP Dataset',
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"Dataset base: {c}")
            return c
    for root, dirs, files in os.walk('/content/drive/MyDrive'):
        depth = root.replace('/content/drive/MyDrive', '').count(os.sep)
        if depth > 4:
            dirs.clear()
            continue
        if 'DeepFashion' in dirs:
            print(f"Dataset base found: {root}")
            return root
    raise FileNotFoundError("Could not find dataset folder. Check Drive is mounted.")


def find_repo_base():
    candidates = [
        '/content/Luxe/backend',
        '/content/luxe/backend',
        '/content/drive/MyDrive/Luxe/backend',
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"Repo base: {c}")
            return c
    raise FileNotFoundError("Luxe repo not found. Run: git clone https://github.com/nabirakhan/luxe /content/Luxe")


DRIVE_BASE = find_drive_base()
REPO_BASE  = find_repo_base()

sys.path.insert(0, REPO_BASE)

import logging
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

if not torch.cuda.is_available():
    raise RuntimeError("No GPU — switch Colab runtime to T4")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PARTITION_FILE = f"{DRIVE_BASE}/DeepFashion/list_eval_partition.txt"
PAIRS_DIR      = Path(f"{DRIVE_BASE}/unet_pairs")
PAIRS_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_CKPT_LOG = Path(f"{DRIVE_BASE}/pair_progress.txt")

IMG_ROOT = "/content/deepfashion/img"


def load_image_list(partition_file: str) -> list[str]:
    images = []
    with open(partition_file) as f:
        # Skip count line and header line
        lines = f.readlines()[2:]
    for line in lines:
        parts = line.strip().split()
        # File format: image_name  item_id  evaluation_status
        if len(parts) >= 3 and parts[2] == "train":
            images.append(parts[0])
    return images


def main():
    from protect import Protector
    protector = Protector()

    image_names = load_image_list(PARTITION_FILE)
    logger.info(f"Found {len(image_names)} train images.")

    generated = 0
    skipped   = 0

    pbar = tqdm(enumerate(image_names), total=len(image_names), desc="Generating pairs")
    for idx, img_name in pbar:
        stem     = Path(img_name).stem
        out_path = PAIRS_DIR / f"{stem}.pt"

        if out_path.exists():
            skipped += 1
            pbar.set_postfix({"generated": generated, "skipped": skipped})
            continue

        img_path = Path(IMG_ROOT) / img_name
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue

        try:
            with open(img_path, "rb") as f:
                image_bytes = f.read()

            x_orig, delta, mask = protector.protect_pgd_only(image_bytes, mode="full")
            # x_orig: [3,512,512], delta: [3,512,512] — batch dim already stripped

            torch.save({"x_orig": x_orig, "delta": delta, "mask": mask}, out_path)
            generated += 1
            pbar.set_postfix({"generated": generated, "skipped": skipped})

            if generated % 10 == 0:
                logger.info(f"Generated {generated} pairs ({skipped} skipped, {idx+1}/{len(image_names)} processed)")

            # Drive checkpoint log every 100 pairs
            if generated % 100 == 0:
                with open(DRIVE_CKPT_LOG, "a") as f:
                    f.write(f"generated={generated} idx={idx}\n")
                logger.info(f"Checkpoint logged at {generated} pairs.")

        except Exception as e:
            logger.error(f"Failed on {img_name}: {e}")
            continue

    logger.info(f"Done. Generated={generated}, skipped={skipped}.")


if __name__ == "__main__":
    main()
