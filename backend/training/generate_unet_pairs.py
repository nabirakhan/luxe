"""Generate (image, pgd_delta) pairs for U-Net distillation training.

Mode is always "full" — trains the U-Net on combined nudify+modify protection.
Runtime: ~92s/image on P100 -> ~77 hours for 3000 pairs across 3 Kaggle sessions.

Resume safety:
- Saves pair_gen_resume.json to Drive after every single pair.
- On restart, reads resume.json and skips straight to where it left off.
- Also skips any .pt file that already exists (double safety net).
- Safe to split across multiple Kaggle accounts — all share the same Drive.

See bottom of file for full team workflow instructions.
"""

import os, sys, json, logging, zipfile
from pathlib import Path
from datetime import datetime

import torch
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Path resolution ───────────────────────────────────────────────────────────
def find_drive_base():
    candidates = [
        '/root/drive/MyDrive/dlpp/DLP Project Datasets',    # Kaggle + Drive addon
        '/root/drive/MyDrive/DLP Project Datasets',
        '/content/drive/MyDrive/dlpp/DLP Project Datasets', # Colab
        '/content/drive/MyDrive/DLP Project Datasets',
        '/content/drive/MyDrive/DLP Dataset',
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"✅ Dataset base: {c}")
            return c
    for root, dirs, files in os.walk('/root/drive/MyDrive'):
        depth = root.replace('/root/drive/MyDrive', '').count(os.sep)
        if depth > 4:
            dirs.clear()
            continue
        if 'DeepFashion' in dirs:
            print(f"✅ Dataset base found: {root}")
            return root
    raise FileNotFoundError(
        "Could not find dataset folder.\n"
        "Make sure Google Drive is mounted and the DLP Project Datasets folder exists.\n"
        "Kaggle: Add-ons → Google Drive → sign in with the shared Drive account."
    )


def find_repo_base():
    candidates = [
        '/kaggle/working/Luxe/backend',
        '/kaggle/working/luxe/backend',
        '/content/Luxe/backend',
        '/content/luxe/backend',
        '/root/drive/MyDrive/Luxe/backend',
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"✅ Repo base: {c}")
            return c
    raise FileNotFoundError(
        "Luxe repo not found.\n"
        "Run: !git clone https://github.com/nabirakhan/luxe /kaggle/working/Luxe"
    )


DRIVE_BASE = find_drive_base()
REPO_BASE  = find_repo_base()
sys.path.insert(0, REPO_BASE)

if not torch.cuda.is_available():
    raise RuntimeError("No GPU — enable GPU accelerator in Kaggle settings (P100)")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")


# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_PAIRS   = 3000
PAIRS_DIR      = Path(DRIVE_BASE) / "unet_pairs"
RESUME_PATH    = Path(DRIVE_BASE) / "pair_gen_resume.json"
CKPT_LOG_PATH  = Path(DRIVE_BASE) / "pair_progress.txt"
PARTITION_FILE = f"{DRIVE_BASE}/DeepFashion/list_eval_partition.txt"
IMG_ROOT       = "/kaggle/working/deepfashion/img"   # local fast SSD, not Drive

PAIRS_DIR.mkdir(parents=True, exist_ok=True)
print(f"✅ Pairs dir  : {PAIRS_DIR}")
print(f"✅ Resume file: {RESUME_PATH}")


# ── Unzip images to local fast storage ───────────────────────────────────────
def unzip_if_needed(zip_path, extract_to, check_subdir):
    if not os.path.exists(os.path.join(extract_to, check_subdir)):
        os.makedirs(extract_to, exist_ok=True)
        logger.info(f"Unzipping {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        logger.info(f"Done → {extract_to}")
    else:
        logger.info(f"Already extracted: {extract_to}")

unzip_if_needed(
    zip_path     = f"{DRIVE_BASE}/DeepFashion/Img/img.zip",
    extract_to   = "/kaggle/working/deepfashion/img/",
    check_subdir = "img",
)


# ── Resume helpers ────────────────────────────────────────────────────────────
def load_resume() -> dict:
    """Load shared resume state from Drive. Defaults if no file exists."""
    if RESUME_PATH.exists():
        with open(RESUME_PATH) as f:
            state = json.load(f)
        logger.info(
            f"🔄 Resume found — "
            f"last_idx={state['last_idx']} | "
            f"total_generated={state['total_generated']} | "
            f"last_run={state.get('last_run','?')} | "
            f"last_account={state.get('last_account','?')}"
        )
        return state
    logger.info("🆕 No resume file — starting fresh")
    return {
        "last_idx":        -1,
        "total_generated":  0,
        "last_run":        None,
        "last_account":    None,
    }


def save_resume(state: dict):
    """Atomic write to Drive — avoids corrupt JSON on mid-write disconnect."""
    state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tmp = RESUME_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(RESUME_PATH)  # atomic rename


# ── Image list ────────────────────────────────────────────────────────────────
def load_image_list(partition_file: str) -> list:
    with open(partition_file) as f:
        lines = f.readlines()[2:]   # skip count + header lines
    images = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[2] == "train":
            images.append(parts[0])
    return images


def count_existing_pairs() -> int:
    return len([f for f in PAIRS_DIR.iterdir() if f.suffix == ".pt"])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── SET YOUR NAME BEFORE RUNNING ──────────────────────────────────────────
    # Nabira → "nabira" | Rameen → "rameen" | Aisha → "aisha"
    ACCOUNT_NAME = "rameen"   # ← CHANGE THIS

    from protect import Protector
    protector = Protector()

    image_names = load_image_list(PARTITION_FILE)
    logger.info(f"Train images in partition: {len(image_names)}")

    state = load_resume()

    # Verify .pt count matches resume state — Drive is source of truth
    actual_count = count_existing_pairs()
    if actual_count != state["total_generated"]:
        logger.warning(
            f"Resume says {state['total_generated']} pairs but "
            f"Drive has {actual_count} .pt files. Using Drive count."
        )
        state["total_generated"] = actual_count

    if state["total_generated"] >= TARGET_PAIRS:
        logger.info(
            f"✅ {state['total_generated']} pairs already on Drive. "
            f"Target of {TARGET_PAIRS} reached. Nothing to do."
        )
        return

    logger.info(
        f"Resuming from image index {state['last_idx'] + 1} | "
        f"{state['total_generated']} on Drive | "
        f"{TARGET_PAIRS - state['total_generated']} remaining"
    )

    generated_this_session = 0
    skipped_this_session   = 0

    pbar = tqdm(
        enumerate(image_names),
        total=len(image_names),
        desc=f"[{ACCOUNT_NAME}] pairs",
    )

    for idx, img_name in pbar:

        # Skip everything already processed in previous sessions
        if idx <= state["last_idx"]:
            skipped_this_session += 1
            continue

        stem     = Path(img_name).stem
        out_path = PAIRS_DIR / f"{stem}.pt"

        # Double safety: skip if .pt exists even if idx wasn't recorded
        if out_path.exists():
            state["last_idx"] = idx
            save_resume(state)
            skipped_this_session += 1
            pbar.set_postfix({
                "session": generated_this_session,
                "total":   state["total_generated"],
            })
            continue

        img_path = Path(IMG_ROOT) / img_name
        if not img_path.exists():
            logger.warning(f"Image not found, skipping: {img_path}")
            state["last_idx"] = idx
            save_resume(state)
            continue

        try:
            with open(img_path, "rb") as f:
                image_bytes = f.read()

            x_orig, delta, mask = protector.protect_pgd_only(image_bytes, mode="full")
            # x_orig: [3,512,512]  delta: [3,512,512]  mask: [1,512,512]
            # Batch dim already stripped in protect_pgd_only

            torch.save({"x_orig": x_orig, "delta": delta, "mask": mask}, out_path)

            # Update and save resume state after every successful pair
            generated_this_session   += 1
            state["total_generated"] += 1
            state["last_idx"]         = idx
            state["last_account"]     = ACCOUNT_NAME
            save_resume(state)

            pbar.set_postfix({
                "session":   generated_this_session,
                "total":     state["total_generated"],
                "remaining": TARGET_PAIRS - state["total_generated"],
            })

            if generated_this_session % 10 == 0:
                logger.info(
                    f"[{ACCOUNT_NAME}] session={generated_this_session} | "
                    f"total={state['total_generated']}/{TARGET_PAIRS}"
                )

            # Append to progress log every 100 pairs
            if generated_this_session % 100 == 0:
                with open(CKPT_LOG_PATH, "a") as f:
                    f.write(
                        f"{state['last_run']} | "
                        f"account={ACCOUNT_NAME} | "
                        f"session={generated_this_session} | "
                        f"total={state['total_generated']} | "
                        f"idx={idx}\n"
                    )
                logger.info(f"Progress logged — {state['total_generated']} pairs on Drive.")

            # Stop exactly at target
            if state["total_generated"] >= TARGET_PAIRS:
                logger.info(
                    f"✅ TARGET REACHED — {state['total_generated']} pairs on Drive. "
                    f"This session: {generated_this_session}. Stopping."
                )
                break

        except Exception as e:
            logger.error(f"Failed on {img_name} (idx={idx}): {e}")
            # Advance idx so the same broken image is never retried
            state["last_idx"] = idx
            save_resume(state)
            continue

    logger.info(
        f"\n{'='*60}\n"
        f"Session complete [{ACCOUNT_NAME}]\n"
        f"  Generated this session : {generated_this_session}\n"
        f"  Skipped this session   : {skipped_this_session}\n"
        f"  Total on Drive         : {state['total_generated']} / {TARGET_PAIRS}\n"
        f"  Remaining              : {max(0, TARGET_PAIRS - state['total_generated'])}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()


# ════════════════════════════════════════════════════════════════════════════════
# TEAM WORKFLOW — SPLITTING ACROSS 3 KAGGLE ACCOUNTS
# ════════════════════════════════════════════════════════════════════════════════
#
# ONE-TIME SETUP
# ──────────────
# 1. Nabira shares the Drive folder with Rameen and Aisha:
#       Google Drive → right-click "DLP Project Datasets" → Share
#       Add both emails → set to Editor
#
# 2. Each person connects Drive to their Kaggle:
#       Kaggle notebook → Add-ons (top menu) → Google Drive
#       Sign in with your Google account (the one that has access to shared folder)
#       Drive mounts at /root/drive/MyDrive
#       Verify: !ls "/root/drive/MyDrive/dlpp/DLP Project Datasets/"
#
# 3. Each person clones the repo at the start of every Kaggle session:
#       !git clone https://github.com/nabirakhan/luxe /kaggle/working/Luxe
#
# EVERY SESSION
# ─────────────
# 1. Change ACCOUNT_NAME in main() to your name before running
# 2. Run the script
# 3. It reads pair_gen_resume.json from the shared Drive
#    and picks up exactly where the last person stopped
# 4. When Kaggle disconnects, the resume file is already saved on Drive
#    after every single pair — just rerun the script to continue
#
# HANDOFF BETWEEN ACCOUNTS
# ─────────────────────────
# No coordination needed. Whoever opens Kaggle next just runs the script.
# The shared resume.json on Drive tracks the exact position automatically.
#
# CHECK PROGRESS ANYTIME
# ───────────────────────
# Paste this in any Kaggle cell:
#
#   import json
#   path = "/root/drive/MyDrive/dlpp/DLP Project Datasets/pair_gen_resume.json"
#   with open(path) as f:
#       print(json.dumps(json.load(f), indent=2))
#
# Or check the full log:
#   !tail -20 "/root/drive/MyDrive/dlpp/DLP Project Datasets/pair_progress.txt"
#
# ESTIMATED TIMELINE (92s/image, 30hr Kaggle sessions)
# ──────────────────────────────────────────────────────
#   Session 1 (~30hr) : pairs    1 – 1173  (any account)
#   Session 2 (~30hr) : pairs 1174 – 2346  (any account)
#   Session 3 (~17hr) : pairs 2347 – 3000  (stops automatically)
#
# Sessions can be split mid-way between accounts — handoff is seamless.
# Example: Rameen does 600, Nabira does 573, Aisha does the rest of session 1.
# ════════════════════════════════════════════════════════════════════════════════
