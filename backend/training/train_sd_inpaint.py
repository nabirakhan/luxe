"""Fine-tune SD v1.5 inpainting UNet on DeepFashion.

Standard diffusion fine-tuning (NOT DreamBooth — DreamBooth is for 3-30 images).
Fine-tunes cross-attention and self-attention layers in the inpainting UNet
on DeepFashion images with clothing masks applied as inpainting masks.

Domain-specific fine-tuning on DeepFashion without prior preservation causes
catastrophic forgetting of non-clothing inpainting (faces, backgrounds).
This is intentional — it strengthens the white-box surrogate for clothing attack.
Grey-box transfer to vanilla SD v2/SDXL may be weaker as a result; grey-box
results reported honestly in eval.

Hyperparameters: AdamW lr=1e-5, 5 epochs, fp16 loading + fp32 attention grads.
Run on Colab T4. Unzip cell at top of session.
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
from diffusers import StableDiffusionInpaintPipeline
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler

from data.deepfashion_loader import DeepFashionDataset

if not torch.cuda.is_available():
    raise RuntimeError("No GPU — switch Colab runtime to T4")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

DRIVE_CKPT_DIR = Path(f"{DRIVE_BASE}/checkpoints")
DRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)

RESUME_PATH    = DRIVE_CKPT_DIR / "sd_inpaint_resume.pth"
PARTITION_FILE = f"{DRIVE_BASE}/DeepFashion/list_eval_partition.txt"

IMG_ROOT   = "/content/deepfashion/img/img"
SEG_ROOT   = "/content/deepfashion/seg/img_highres"
EPOCHS     = 5
LR         = 1e-5
BATCH_SIZE = 2
device     = "cuda"


def get_attention_params(unet):
    """Return only attention params, cast to fp32 for stable gradient flow."""
    params = []
    for name, p in unet.named_parameters():
        if "attn" in name:
            p.data = p.data.float()  # cast to fp32
            p.requires_grad_(True)
            params.append(p)
        else:
            p.requires_grad_(False)
    return params


def train():
    print("Loading SD v1.5 inpainting pipeline (fp16)...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    unet            = pipe.unet.to(device)
    vae             = pipe.vae.to(device)
    noise_scheduler = pipe.scheduler

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Attention params are cast to fp32 inside get_attention_params
    attn_params = get_attention_params(unet)
    optimizer   = AdamW(attn_params, lr=LR)
    scaler      = GradScaler()

    dataset = DeepFashionDataset(IMG_ROOT, SEG_ROOT, PARTITION_FILE, split="train")
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Compute null text embedding once
    enc      = pipe.text_encoder.to(device)
    null_ids = pipe.tokenizer(
        "", return_tensors="pt", padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
    ).input_ids.to(device)
    with torch.no_grad():
        encoder_hidden = enc(null_ids).last_hidden_state

    start_epoch = 0
    best_loss   = float("inf")

    if RESUME_PATH.exists():
        ckpt = torch.load(RESUME_PATH, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        unet.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        best_loss = ckpt["best_loss"]
        print(f"Resumed from epoch {ckpt['epoch'] + 1}, best_loss={best_loss:.4f}")

    for epoch in range(start_epoch, EPOCHS):
        unet.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for step, (imgs, masks) in enumerate(pbar):
            imgs  = imgs.to(device, dtype=torch.float16)
            masks = masks.to(device, dtype=torch.float16)

            with torch.no_grad():
                latents = vae.encode(imgs * 2 - 1).latent_dist.sample() * vae.config.scaling_factor

            noise     = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            masked_imgs    = imgs * (1 - masks)
            masked_latents = vae.encode(masked_imgs * 2 - 1).latent_dist.sample() * vae.config.scaling_factor
            mask_resized   = torch.nn.functional.interpolate(
                masks, size=latents.shape[-2:], mode="nearest"
            )

            model_input = torch.cat([noisy_latents, mask_resized, masked_latents], dim=1)
            hidden      = encoder_hidden.expand(imgs.shape[0], -1, -1)

            # autocast handles fp16/fp32 mixed precision correctly
            # attention params are fp32, rest is fp16 — no conflict
            with autocast():
                pred_noise = unet(model_input, timesteps, encoder_hidden_states=hidden).sample
                loss       = torch.nn.functional.mse_loss(
                    pred_noise.float(), noise.float()  # compare in fp32
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if (step + 1) % 500 == 0:
                torch.save({
                    "epoch":     epoch,
                    "model":     unet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler":    scaler.state_dict(),
                    "best_loss": best_loss,
                }, RESUME_PATH)
                print(f"  Mid-epoch checkpoint saved (step {step+1})")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS}  avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

        torch.save({
            "epoch":     epoch,
            "model":     unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler":    scaler.state_dict(),
            "best_loss": best_loss,
        }, RESUME_PATH)

        ckpt_path = DRIVE_CKPT_DIR / f"sd_inpaint_unet_epoch{epoch+1}.pth"
        torch.save(unet.state_dict(), ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}")

    vae_path = DRIVE_CKPT_DIR / "sd_inpaint_vae.pth"
    torch.save(vae.state_dict(), vae_path)
    print(f"VAE saved → {vae_path}")
    print("SD inpainting fine-tuning complete.")


train()