"""Train SegFormer-B2 with a 20-class LIP head.

Run on Colab T4. Mount Drive and unzip datasets first (see unzip cell below).

Unzip cell (run once per session):
-----------------------------------------------------------------------
import zipfile, os

datasets = [
    ('/content/drive/MyDrive/DLP Dataset/DeepFashion/Img/img.zip', '/content/deepfashion/img/'),
    ('/content/drive/MyDrive/DLP Dataset/DeepFashion/img_highres_seg.zip', '/content/deepfashion/seg/'),
    ('/content/drive/MyDrive/DLP Dataset/Look Into Person/TrainVal_images/TrainVal_images.zip', '/content/lip/images/'),
    ('/content/drive/MyDrive/DLP Dataset/Look Into Person/TrainVal_parsing_annotations/TrainVal_parsing_annotations/TrainVal_parsing_annotations.zip', '/content/lip/annotations/'),
]
for zip_path, extract_to in datasets:
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        print(f"Done → {extract_to}")
    else:
        print(f"Already extracted: {extract_to}")
-----------------------------------------------------------------------

Hyperparameters: AdamW lr=6e-5, cosine LR schedule, 20 epochs.
Saves checkpoints to Drive after each epoch.
"""

import os
import sys
sys.path.insert(0, "/content/drive/MyDrive/Luxe/backend")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch.nn.functional as F
from pathlib import Path

from data.lip_loader import get_lip_loaders

DRIVE_CKPT_DIR = Path("/content/drive/MyDrive/Luxe/checkpoints")
DRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)

LIP_IMAGES = "/content/lip/images"
LIP_ANNS   = "/content/lip/annotations"
NUM_LABELS = 20
EPOCHS     = 20
LR         = 6e-5
BATCH_SIZE = 4
device     = "cuda" if torch.cuda.is_available() else "cpu"


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

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.squeeze(1).to(device)  # [B, H, W] long

            outputs = model(pixel_values=imgs)
            logits  = outputs.logits  # [B, num_labels, H', W']
            logits_up = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)

            loss = F.cross_entropy(logits_up, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.squeeze(1).to(device)
                logits = model(pixel_values=imgs).logits
                logits_up = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
                val_loss += F.cross_entropy(logits_up, labels).item()

        avg_train = total_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}  train={avg_train:.4f}  val={avg_val:.4f}")

        # Save to Drive
        ckpt_path = DRIVE_CKPT_DIR / f"segformer_lip_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), DRIVE_CKPT_DIR / "segformer_lip.pth")
            print(f"  → Best model saved (val={avg_val:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    train()
