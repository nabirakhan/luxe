"""Train SegFormer-B2 with a 20-class LIP head.

Run on Colab T4. Mount Drive and unzip LIP datasets first.

Unzip cell (run once per session):
-----------------------------------------------------------------------
import zipfile, os
datasets = [
    ('/content/drive/MyDrive/Datasets/DLP Project Datasets/Look Into Person/TrainVal_images/TrainVal_images.zip', '/content/lip/images/'),
    ('/content/drive/MyDrive/Datasets/DLP Project Datasets/Look Into Person/TrainVal_parsing_annotations/TrainVal_parsing_annotations.zip', '/content/lip/annotations/'),
]
for zip_path, extract_to in datasets:
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        print(f"Done -> {extract_to}")
    else:
        print(f"Already extracted: {extract_to}")
-----------------------------------------------------------------------

Hyperparameters: AdamW lr=6e-5, cosine LR schedule, 20 epochs.
Saves checkpoints to Drive after each epoch. Resume-safe.
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
        '/content/drive/MyDrive/DLP_Project/DLP Project Datasets',
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

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm

from data.lip_loader import get_lip_loaders

if not torch.cuda.is_available():
    raise RuntimeError("No GPU — switch Colab runtime to T4")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

DRIVE_CKPT_DIR = Path(f"{DRIVE_BASE}/checkpoints")
DRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)

RESUME_PATH = DRIVE_CKPT_DIR / "segformer_resume.pth"

LIP_IMAGES = "/content/lip/images"
LIP_ANNS   = "/content/lip/annotations"
NUM_LABELS = 20
EPOCHS     = 20
LR         = 6e-5
BATCH_SIZE = 4
device     = "cuda"


def compute_miou(logits_up, labels, num_classes=20):
    preds = logits_up.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        pred_c  = (preds == cls)
        label_c = (labels == cls)
        intersection = (pred_c & label_c).sum().float()
        union        = (pred_c | label_c).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    return sum(ious) / len(ious) if ious else 0.0


def train():
    train_loader, val_loader = get_lip_loaders(LIP_IMAGES, LIP_ANNS, batch_size=BATCH_SIZE)

    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler()

    start_epoch   = 0
    best_val_loss = float("inf")

    if RESUME_PATH.exists():
        ckpt = torch.load(RESUME_PATH, map_location=device)
        start_epoch   = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        best_val_loss = ckpt["best_loss"]
        print(f"Resumed from epoch {ckpt['epoch'] + 1}, best_loss={best_val_loss:.4f}")
    else:
        print("No resume checkpoint — starting fresh")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for step, (imgs, labels) in enumerate(pbar):
            imgs   = imgs.to(device)
            labels = labels.squeeze(1).to(device)

            with autocast():
                outputs   = model(pixel_values=imgs)
                logits    = outputs.logits
                logits_up = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
                loss      = F.cross_entropy(logits_up, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if (step + 1) % 500 == 0:
                torch.save({
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler":    scaler.state_dict(),
                    "best_loss": best_val_loss,
                }, RESUME_PATH)
                print(f"  Mid-epoch checkpoint saved (step {step+1})")

        scheduler.step()

        # Validation
        model.eval()
        val_loss    = 0.0
        miou_scores = []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validation", leave=False):
                imgs   = imgs.to(device)
                labels = labels.squeeze(1).to(device)
                with autocast():
                    logits    = model(pixel_values=imgs).logits
                    logits_up = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
                val_loss += F.cross_entropy(logits_up, labels).item()
                miou_scores.append(compute_miou(logits_up, labels))

        avg_train = total_loss  / len(train_loader)
        avg_val   = val_loss    / len(val_loader)
        avg_miou  = sum(miou_scores) / len(miou_scores) if miou_scores else 0.0
        print(f"Epoch {epoch+1}/{EPOCHS}  train={avg_train:.4f}  val={avg_val:.4f}  mIoU={avg_miou:.4f}")

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), DRIVE_CKPT_DIR / "segformer_lip.pth")
            print(f"  Best model saved (val={avg_val:.4f}  mIoU={avg_miou:.4f})")

        # End-of-epoch resume checkpoint with updated best_val_loss
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict(),
            "best_loss": best_val_loss,
        }, RESUME_PATH)

        # Per-epoch checkpoint
        torch.save(model.state_dict(), DRIVE_CKPT_DIR / f"segformer_lip_epoch{epoch+1}.pth")

    print("Training complete.")


train()