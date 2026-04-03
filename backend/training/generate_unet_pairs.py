"""Generate (image, pgd_delta) pairs for U-Net distillation training.

Mode is always "full" — trains the U-Net on combined nudify+modify protection.
Runtime: ~92s/image on T4/P100 -> ~77 hours for 3000 pairs. Plan 3 sessions.

Features:
- Resume-safe: skips images where unet_pairs/{stem}.pt already exists.
- Per-image error handling: logs failures and continues (never crashes the run).
- Drive checkpoint after every 100 pairs.
- Strips batch dim before saving: DataLoader expects [3,512,512] not [1,3,512,512].
- Hardcoded TARGET_PAIRS=3000 stop — safe to split across multiple accounts/sessions.

Run on Kaggle P100 or Colab T4. Mount Google Drive before running.
"""

import os
import sys
import logging
from pathlib import Path

# ── Step 1: Mount Google Drive ───────────────────────────────────────────────
# Kaggle: Drive is mounted via kaggle secrets + gcsfuse or manually.
# Colab:  Uncomment the block below.

# from google.colab import drive
# drive.mount('/content/drive')

# ── Step 2: Unzip datasets if needed ────────────────────────────────────────
import zipfile

def unzip_if_needed(zip_path, extract_to, check_subdir):
    if not os.path.exists(os.path.join(extract_to, check_subdir)):
        os.makedirs(extract_to, exist_ok=True)
        print(f"Unzipping {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        print(f"Done → {extract_to}")
    else:
        print(f"Already extracted: {extract_to}")

# ── Step 3: Path resolution ──────────────────────────────────────────────────
def find_drive_base():
    candidates = [
        '/content/drive/MyDrive/Datasets/DLP Project Datasets',
        '/content/drive/MyDrive/DLP Project Datasets',
        '/content/drive/MyDrive/DLP Dataset',
        '/kaggle/input/dlp-project-datasets',   # Kaggle dataset path
        '/kaggle/input/dlp-datasets',
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
        '/kaggle/working/Luxe/backend',         # Kaggle working dir
        '/kaggle/working/luxe/backend',
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"Repo base: {c}")
            return c
    raise FileNotFoundError(
        "Luxe repo not found.\n"
        "Colab:  git clone https://github.com/nabirakhan/luxe /content/Luxe\n"
        "Kaggle: git clone https://github.com/nabirakhan/luxe /kaggle/working/Luxe"
    )


# ── Output dir for pairs — writable on both platforms ───────────────────────
def find_pairs_output_dir(drive_base):
    # On Kaggle, /kaggle/working is writable but ephemeral.
    # Save pairs to Drive so they persist across sessions.
    # If Drive is not mounted on Kaggle, pairs go to /kaggle/working/unet_pairs
    # and must be manually copied to Drive after each session.
    drive_pairs = Path(drive_base) / "unet_pairs"
    try:
        drive_pairs.mkdir(parents=True, exist_ok=True)
        # Quick write test
        test = drive_pairs / ".write_test"
        test.touch()
        test.unlink()
        print(f"Pairs output: {drive_pairs} (Drive — persistent)")
        return drive_pairs
    except Exception:
        fallback = Path("/kaggle/working/unet_pairs")
        fallback.mkdir(parents=True, exist_ok=True)
        print(
            f"⚠️  Drive not writable. Pairs output: {fallback}\n"
            f"   Copy to Drive after each session to avoid losing progress."
        )
        return fallback


DRIVE_BASE = find_drive_base()
REPO_BASE  = find_repo_base()
sys.path.insert(0, REPO_BASE)

import torch
from tqdm.auto import tqdm

if not torch.cuda.is_available():
    raise RuntimeError("No GPU — switch runtime to GPU (P100 on Kaggle, T4 on Colab)")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_PAIRS   = 3000   # stop after this many pairs — safe across split sessions
CHECKPOINT_LOG = Path(DRIVE_BASE) / "pair_progress.txt"
PAIRS_DIR      = find_pairs_output_dir(DRIVE_BASE)
PARTITION_FILE = f"{DRIVE_BASE}/DeepFashion/list_eval_partition.txt"
IMG_ROOT       = "/content/deepfashion/img"   # local unzipped path

# Unzip DeepFashion images if not already done
unzip_if_needed(
    zip_path    = f"{DRIVE_BASE}/DeepFashion/Img/img.zip",
    extract_to  = "/content/deepfashion/img/",
    check_subdir= "img",
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_image_list(partition_file: str) -> list:
    images = []
    with open(partition_file) as f:
        lines = f.readlines()[2:]   # skip count line + header line
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[2] == "train":
            images.append(parts[0])
    return images


def count_existing_pairs() -> int:
    return len([f for f in PAIRS_DIR.iterdir() if f.suffix == ".pt"])


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    from protect import Protector
    protector = Protector()

    image_names = load_image_list(PARTITION_FILE)
    logger.info(f"Found {len(image_names)} train images.")

    # Count pairs already on Drive (from previous sessions / other account)
    existing = count_existing_pairs()
    logger.info(f"Pairs already generated (this Drive): {existing}")

    if existing >= TARGET_PAIRS:
        logger.info(f"✅ Already have {existing} pairs — target reached. Nothing to do.")
        return

    generated = 0
    skipped   = 0

    pbar = tqdm(enumerate(image_names), total=len(image_names), desc="Generating pairs")
    for idx, img_name in pbar:
        stem     = Path(img_name).stem
        out_path = PAIRS_DIR / f"{stem}.pt"

        # Resume-safe: skip if already generated in any previous session
        if out_path.exists():
            skipped += 1
            pbar.set_postfix({"generated": generated, "skipped": skipped, "on_drive": existing + generated})
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
            pbar.set_postfix({"generated": generated, "skipped": skipped, "on_drive": existing + generated})

            if generated % 10 == 0:
                logger.info(
                    f"Generated {generated} pairs this session "
                    f"({existing + generated} total on Drive, {skipped} skipped)"
                )

            # Checkpoint log to Drive every 100 pairs
            if generated % 100 == 0:
                with open(CHECKPOINT_LOG, "a") as f:
                    f.write(f"session_generated={generated} total_on_drive={existing + generated} idx={idx}\n")
                logger.info(f"Checkpoint logged at {generated} pairs this session.")

            # ── STOP when target reached (safe across split sessions/accounts) ──
            if existing + generated >= TARGET_PAIRS:
                logger.info(
                    f"✅ Target of {TARGET_PAIRS} pairs reached "
                    f"({generated} this session + {existing} from previous). Stopping."
                )
                break

        except Exception as e:
            logger.error(f"Failed on {img_name}: {e}")
            continue

    logger.info(
        f"Session complete. Generated={generated}, skipped={skipped}, "
        f"total on Drive={existing + generated}."
    )


if __name__ == "__main__":
    main()
