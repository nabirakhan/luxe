"""Train CloakUNet to distil PGD adversarial deltas into a fast feed-forward model.

Run on Colab T4 after generate_unet_pairs.py has produced pairs.

Setup order (CRITICAL): pip install lpips BEFORE this training cell.
lpips.LPIPS(net='alex') downloads AlexNet weights on first call — if this happens
inside the training loop, a failed download mid-run loses all progress.

Hyperparameters:
    AdamW lr=1e-4, CosineAnnealingLR T_max=30, batch=8, 30 epochs, patience=5.

Loss:
    total = adv_loss / α_ema + LAMBDA1 * (lpips_loss / β_ema)
    where α/β are EMA-normalised for scale-invariant weighting.
    EMA updated AFTER computing loss (updating before creates self-reference bias).

LPIPS inputs rescaled to [-1, +1] as required by the lpips library.
Gradient clipping: max_norm=1.0.
"""

import sys
sys.path.insert(0, "/content/drive/MyDrive/Luxe/backend")

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

import config

PAIRS_DIR      = Path("/content/drive/MyDrive/Luxe/data/unet_pairs")
DRIVE_CKPT_DIR = Path("/content/drive/MyDrive/Luxe/checkpoints")
DRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS     = 30
LR         = 1e-4
BATCH_SIZE = 8
PATIENCE   = 5
device     = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class PairDataset(Dataset):
    def __init__(self, pairs_dir: Path):
        self.files = sorted(pairs_dir.glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        return data["x_orig"], data["delta"]  # both [3, 512, 512]


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train():
    # LPIPS init at top with error message — must succeed before training loop
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device)
        lpips_model.eval()
        for p in lpips_model.parameters():
            p.requires_grad_(False)
    except Exception as e:
        raise RuntimeError(
            "AlexNet weights for LPIPS could not be downloaded. Check internet connectivity."
        ) from e

    from unet import CloakUNet
    unet = CloakUNet().to(device)

    dataset = PairDataset(PAIRS_DIR)
    if len(dataset) == 0:
        raise RuntimeError(f"No pair files found in {PAIRS_DIR}. Run generate_unet_pairs.py first.")
    print(f"Dataset: {len(dataset)} pairs.")

    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = AdamW(unet.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # EMA normalisers — realistic priors for MSE in [-EPS_PGD, +EPS_PGD] range
    alpha_ema = 0.001   # adv_loss prior
    beta_ema  = 0.01    # lpips_loss prior

    best_val_loss = float("inf")
    patience_cnt  = 0

    for epoch in range(EPOCHS):
        unet.train()
        total_loss = 0.0

        for x_orig, pgd_delta in loader:
            x_orig    = x_orig.to(device)
            pgd_delta = pgd_delta.to(device)

            unet_delta = unet(x_orig)                           # [B, 3, 512, 512]
            adv_loss   = F.mse_loss(unet_delta, pgd_delta)      # regression target

            # LPIPS: inputs must be in [-1, +1]
            x_in  = x_orig              * 2 - 1
            x_out = (x_orig + unet_delta) * 2 - 1
            lpips_loss = lpips_model(x_in, x_out).mean()

            # Normalise by PREVIOUS EMA then update
            total_step_loss = adv_loss / alpha_ema + config.LAMBDA1 * (lpips_loss / beta_ema)

            alpha_ema = 0.99 * alpha_ema + 0.01 * adv_loss.item()
            beta_ema  = 0.99 * beta_ema  + 0.01 * lpips_loss.item()

            optimizer.zero_grad()
            total_step_loss.backward()
            nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += total_step_loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.4f}  α_ema={alpha_ema:.5f}  β_ema={beta_ema:.5f}")

        # Save checkpoint every epoch to Drive
        ckpt = DRIVE_CKPT_DIR / f"cloak_unet_epoch{epoch+1}.pth"
        torch.save(unet.state_dict(), ckpt)

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save(unet.state_dict(), DRIVE_CKPT_DIR / "cloak_unet.pth")
            print(f"  → Best model saved.")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print("U-Net training complete.")


if __name__ == "__main__":
    train()
