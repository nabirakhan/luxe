"""Fine-tune SD v1.5 inpainting UNet on DeepFashion.

Standard diffusion fine-tuning (NOT DreamBooth — DreamBooth is for 3-30 images).
Fine-tunes cross-attention and self-attention layers in the inpainting UNet
on DeepFashion images with clothing masks applied as inpainting masks.

Domain-specific fine-tuning on DeepFashion without prior preservation causes
catastrophic forgetting of non-clothing inpainting (faces, backgrounds).
This is intentional — it strengthens the white-box surrogate for clothing attack.
Grey-box transfer to vanilla SD v2/SDXL may be weaker as a result; grey-box
results reported honestly in eval.

Hyperparameters: AdamW lr=1e-5, 5 epochs, fp16.
Run on Colab T4. Unzip cell at top of session (see train_segformer.py).
"""

import sys
sys.path.insert(0, "/content/drive/MyDrive/Luxe/backend")

import os
from pathlib import Path

import torch
from torch.optim import AdamW
from diffusers import StableDiffusionInpaintPipeline
from accelerate import Accelerator
from torch.utils.data import DataLoader

from data.deepfashion_loader import DeepFashionDataset

DRIVE_CKPT_DIR = Path("/content/drive/MyDrive/Luxe/checkpoints")
DRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)

IMG_ROOT       = "/content/deepfashion/img"
SEG_ROOT       = "/content/deepfashion/seg"
PARTITION_FILE = "/content/deepfashion/list_eval_partition.txt"
EPOCHS         = 5
LR             = 1e-5
BATCH_SIZE     = 2


def get_attention_params(unet):
    """Return only cross-attention and self-attention parameters."""
    params = []
    for name, p in unet.named_parameters():
        if "attn" in name:
            p.requires_grad_(True)
            params.append(p)
        else:
            p.requires_grad_(False)
    return params


def train():
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    print("Loading SD v1.5 inpainting pipeline...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    unet    = pipe.unet.to(device)
    vae     = pipe.vae.to(device)
    noise_scheduler = pipe.scheduler

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    attn_params = get_attention_params(unet)
    optimizer   = AdamW(attn_params, lr=LR)

    dataset = DeepFashionDataset(IMG_ROOT, SEG_ROOT, PARTITION_FILE, split="train")
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)

    for epoch in range(EPOCHS):
        unet.train()
        total_loss = 0.0
        for step, (imgs, masks) in enumerate(loader):
            imgs  = imgs.to(device, dtype=torch.float16)
            masks = masks.to(device, dtype=torch.float16)

            with torch.no_grad():
                latents = vae.encode(imgs * 2 - 1).latent_dist.sample() * vae.config.scaling_factor

            noise     = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Masked latents for inpainting conditioning
            masked_imgs   = imgs * (1 - masks)
            masked_latents = vae.encode(masked_imgs * 2 - 1).latent_dist.sample() * vae.config.scaling_factor
            mask_resized  = torch.nn.functional.interpolate(masks, size=latents.shape[-2:], mode="nearest")

            model_input = torch.cat([noisy_latents, mask_resized, masked_latents], dim=1)

            # Unconditional (null text embedding)
            enc = pipe.text_encoder
            null_ids = pipe.tokenizer("", return_tensors="pt", padding="max_length",
                                      max_length=pipe.tokenizer.model_max_length).input_ids.to(device)
            with torch.no_grad():
                encoder_hidden = enc(null_ids).last_hidden_state

            pred_noise = unet(model_input, timesteps, encoder_hidden_states=encoder_hidden).sample
            loss = torch.nn.functional.mse_loss(pred_noise, noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 100 == 0:
                print(f"  Epoch {epoch+1} step {step}  loss={loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS}  avg_loss={avg_loss:.4f}")

        # Save UNet + VAE to Drive
        ckpt_path = DRIVE_CKPT_DIR / f"sd_inpaint_unet_epoch{epoch+1}.pth"
        torch.save(accelerator.unwrap_model(unet).state_dict(), ckpt_path)

    # Save final VAE (unchanged) for use in InpaintLoss
    vae_path = DRIVE_CKPT_DIR / "sd_inpaint_vae.pth"
    torch.save(vae.state_dict(), vae_path)
    print(f"VAE saved to {vae_path}")
    print("SD inpainting fine-tuning complete.")


if __name__ == "__main__":
    train()
