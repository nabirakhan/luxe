"""Generate (image, pgd_delta) pairs for U-Net distillation training.

Mode is always "full" — trains the U-Net on combined nudify+modify protection.
Runtime: ~5–10s/image on T4 → 14–28 hours for 10k images. Plan 3+ sessions.

Features:
- Resume-safe: skips images where data/unet_pairs/{stem}.pt already exists.
- Per-image error handling: logs failures and continues (never crashes the run).
- Drive checkpoint after every 100 pairs.
- Strips batch dim before saving: DataLoader expects [3,512,512] not [1,3,512,512].

Run on Colab T4. Unzip cell at top of session (see train_segformer.py).
"""

import sys
sys.path.insert(0, "/content/drive/MyDrive/Luxe/backend")

import io
import logging
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMG_ROOT       = "/content/deepfashion/img"
PARTITION_FILE = "/content/deepfashion/list_eval_partition.txt"
PAIRS_DIR      = Path("/content/drive/MyDrive/Luxe/data/unet_pairs")
PAIRS_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_CKPT_LOG = Path("/content/drive/MyDrive/Luxe/data/pair_progress.txt")


def load_image_list(partition_file: str) -> list[str]:
    images = []
    with open(partition_file) as f:
        lines = f.readlines()[2:]
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[1] == "train":
            images.append(parts[0])
    return images


def main():
    from protect import Protector
    protector = Protector()

    image_names = load_image_list(PARTITION_FILE)
    logger.info(f"Found {len(image_names)} train images.")

    generated = 0
    skipped   = 0

    for idx, img_name in enumerate(image_names):
        stem     = Path(img_name).stem
        out_path = PAIRS_DIR / f"{stem}.pt"

        if out_path.exists():
            skipped += 1
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
