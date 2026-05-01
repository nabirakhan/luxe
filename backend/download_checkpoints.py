"""Download model checkpoints from Google Drive.

Run once before starting the server, or add to Render build command:
    pip install -r requirements.txt && python download_checkpoints.py
"""

import os
import sys
from pathlib import Path

import gdown

FOLDER_ID = "18cprgjVtQcEm4B9g-S9N0wpjHziGlUS5"
DEST = Path(__file__).parent / "checkpoints"

REQUIRED = [
    "segformer_lip.pth",
    "cloak_unet.pth",
    "sd_inpaint_vae.pth",
    "ipp_vae.pth",
    "ip_adapter.pth",
]


def main():
    DEST.mkdir(exist_ok=True)

    missing = [f for f in REQUIRED if not (DEST / f).exists()]
    if not missing:
        print("All checkpoints already present — skipping download.")
        return

    print(f"Missing: {missing}")
    print(f"Downloading from Google Drive folder {FOLDER_ID} ...")

    gdown.download_folder(
        id=FOLDER_ID,
        output=str(DEST),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )

    # Verify
    still_missing = [f for f in REQUIRED if not (DEST / f).exists()]
    if still_missing:
        print(f"ERROR: still missing after download: {still_missing}", file=sys.stderr)
        sys.exit(1)

    print("All checkpoints downloaded successfully.")


if __name__ == "__main__":
    main()
