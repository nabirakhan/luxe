"""Fine-tune IP-Adapter (IPP) on paired DeepFashion outfit views.

Hyperparameters: AdamW lr=5e-6, 3 epochs.
Run on Colab T4. Unzip cell at top of session (see train_segformer.py).

Saves checkpoints to Drive. The image_proj_model weights are what
pgd_modification.py loads via ip_adapter.pth.
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

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

if not torch.cuda.is_available():
    raise RuntimeError("No GPU — switch Colab runtime to T4")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

DRIVE_CKPT_DIR = Path(f"{DRIVE_BASE}/checkpoints")
DRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)

RESUME_PATH = DRIVE_CKPT_DIR / "ipp_resume.pth"
INSHOP_FILE = f"{DRIVE_BASE}/DeepFashion/list_item_inshop.txt"

IMG_ROOT   = "/content/deepfashion/img"
EPOCHS     = 3
LR         = 5e-6
BATCH_SIZE = 4
device     = "cuda"


class PairedOutfitDataset(Dataset):
    """Paired outfit views from list_item_inshop.txt."""

    def __init__(self, img_root: str, inshop_file: str):
        self.img_root = Path(img_root)
        self.pairs = []
        with open(inshop_file) as f:
            lines = f.readlines()[2:]
        item_to_images = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_name, item_id = parts[0], parts[1]
            item_to_images.setdefault(item_id, []).append(img_name)
        for item_id, imgs in item_to_images.items():
            for i in range(len(imgs) - 1):
                self.pairs.append((imgs[i], imgs[i + 1]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a_name, b_name = self.pairs[idx]
        def load(name):
            p = self.img_root / name
            img = Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS)
            return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return load(a_name), load(b_name)


def train():
    import clip

    dataset = PairedOutfitDataset(IMG_ROOT, INSHOP_FILE)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # CLIP image encoder (frozen)
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # VAE (frozen) — used to measure latent alignment
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Image projection head: 1024 (ViT-L/14) -> 768
    from torch import nn
    image_proj = nn.Linear(1024, 768, bias=False).to(device)
    optimizer  = AdamW(image_proj.parameters(), lr=LR)
    scaler     = GradScaler()

    from utils import preprocess_for_clip

    start_epoch   = 0
    best_loss     = float("inf")

    if RESUME_PATH.exists():
        ckpt = torch.load(RESUME_PATH, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        image_proj.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        best_loss = ckpt["best_loss"]
        print(f"Resumed from epoch {ckpt['epoch'] + 1}, best_loss={best_loss:.5f}")

    for epoch in range(start_epoch, EPOCHS):
        image_proj.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step, (imgs_a, imgs_b) in enumerate(pbar):
            imgs_a = imgs_a.to(device)
            imgs_b = imgs_b.to(device)

            # CLIP features for source image (a)
            a_pre = preprocess_for_clip(imgs_a)
            with torch.no_grad():
                clip_feat = clip_model.encode_image(a_pre).float()  # [B, 1024]

            with autocast():
                proj = image_proj(clip_feat)  # [B, 768]

                # Target: VAE latent of the paired image (b)
                with torch.no_grad():
                    lat_b      = vae.encode(imgs_b * 2 - 1).latent_dist.mean  # [B, 4, 64, 64]
                    lat_b_pooled = lat_b.mean(dim=[-2, -1])                    # [B, 4]
                    lat_b_rep  = lat_b_pooled.repeat(1, 768 // 4 + 1)[:, :768]
                    lat_b_rep  = torch.nn.functional.normalize(lat_b_rep, dim=-1)

                proj_norm = torch.nn.functional.normalize(proj, dim=-1)
                loss      = torch.nn.functional.mse_loss(proj_norm, lat_b_rep)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

            if (step + 1) % 500 == 0:
                torch.save({
                    "epoch":     epoch,
                    "model":     image_proj.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler":    scaler.state_dict(),
                    "best_loss": best_loss,
                }, RESUME_PATH)

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS}  avg_loss={avg:.5f}")

        # End-of-epoch resume checkpoint
        torch.save({
            "epoch":     epoch,
            "model":     image_proj.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler":    scaler.state_dict(),
            "best_loss": best_loss,
        }, RESUME_PATH)

        if avg < best_loss:
            best_loss = avg

        ckpt = DRIVE_CKPT_DIR / f"ipp_proj_epoch{epoch+1}.pth"
        torch.save({"image_proj_model.weight": image_proj.weight.data}, ckpt)

    # Final save in format expected by pgd_modification.py
    final_path = DRIVE_CKPT_DIR / "ip_adapter.pth"
    torch.save({"image_proj_model.weight": image_proj.weight.data}, final_path)
    print(f"Saved -> {final_path}")

    # Also save VAE weights for ModificationLoss IPP path
    vae_path = DRIVE_CKPT_DIR / "ipp_vae.pth"
    torch.save(vae.state_dict(), vae_path)
    print(f"IPP VAE saved -> {vae_path}")


if __name__ == "__main__":
    train()
