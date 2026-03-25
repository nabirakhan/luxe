# Luxe — AI Image Privacy Protection

> ⚠️ **This project is currently in progress.** Training is actively running. Checkpoints, evaluation results, and final deployment are not yet complete.

Luxe is a free, privacy-first web tool that protects portrait photos from AI nudification and outfit-modification attacks by embedding imperceptible adversarial perturbations. Built as a Deep Learning Practice course project (2026) by Nabira, Rameen, and Aisha.

The core innovation is a trained U-Net that generates protection perturbations in a single forward pass — approximately 90× faster than standard PGD — making it practical for everyday CPU use in under 10 seconds per image.

---

## What It Does

AI tools are increasingly being weaponized to manipulate photos of people — removing clothing, swapping outfits, or generating non-consensual intimate imagery (NCII) from ordinary photos shared online. Luxe fights back by embedding invisible adversarial noise into images before they are shared online.

Two threat classes are addressed simultaneously:

- **Nudification** — AI tools that remove or alter clothing to generate non-consensual intimate imagery
- **Appearance modification** — AI tools that swap outfits, change clothing style, or alter body appearance

---

## Current Status

| Component | Status |
|---|---|
| Frontend (React + Three.js) | ✅ Complete |
| Backend (FastAPI) | ✅ Complete |
| SegFormer-B2 training on LIP | 🔄 In Progress |
| SD v1.5 Inpainting fine-tuning | 🔄 In Progress |
| InstructPix2Pix fine-tuning | 🔄 In Progress |
| UNet pair generation | ⏳ Not Started — depends on above |
| UNet training | ⏳ Not Started — depends on pair generation |
| Evaluation | ⏳ Not Started — depends on all training |
| Deployment (Vercel + Render) | ⏳ Not Started |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite + Three.js + Tailwind CSS |
| Backend | Python 3.10 + FastAPI + Uvicorn |
| Adversarial attacks | PyTorch + PGD + CLIP ensemble |
| Nudification surrogate | Stable Diffusion v1.5 Inpainting (fine-tuned) |
| Modification surrogate | InstructPix2Pix + IP-Adapter |
| Segmentation surrogate | SegFormer-B2 fine-tuned on LIP |
| Protection network | Custom trained U-Net (novel contribution) |
| Training compute | Google Colab T4 GPU |
| Datasets | DeepFashion In-Shop + LIP (Look Into Person) |

---

## Datasets

| Dataset | Used For |
|---|---|
| DeepFashion In-Shop (52,712 images) | SD surrogate training, UNet pair generation, evaluation |
| LIP — Look Into Person (50,462 images) | SegFormer surrogate training |

---

## Evaluation Targets

| Metric | Target |
|---|---|
| LPIPS (original vs cloaked) | < 0.05 |
| SSIM | > 0.95 |
| PSNR | > 35 dB |
| CLIP cosine similarity after cloaking | < 0.50 |
| Nudification disruption LPIPS delta (white-box) | > 0.75 |
| Outfit-swap disruption LPIPS delta (white-box) | > 0.60 |
| UNet inference time | < 10 seconds on CPU |

---

## Academic Scope & Honest Framing

This project is evaluated entirely within the open-source model ecosystem. No primary claims are made about closed proprietary systems such as Grok Aurora, Gemini, or DALL-E 3. This is the accepted standard in the adversarial ML literature — PhotoGuard (MIT, 2023), Glaze (UChicago, 2023), and Anti-DreamBooth (2023) all evaluated exclusively on open-source models.

On grey-box open-source models not used as direct attack targets, partial transfer of 40–70% of white-box effectiveness is the expected and honest result.

---

## Team

- Aisha Asif
- Nabira Khan
- Rameen Zehra
  
---

## Limitations

- Evaluated against open-source surrogate models only — no claims about proprietary systems
- Grey-box transfer to non-surrogate models yields 40–70% of white-box effectiveness
- Heavy recompression (JPEG quality < 50) can degrade perturbations
- Output resolution is fixed at 512×512
- No tool can guarantee protection against all future AI architectures

---

*Deep Learning Practice Course Project · FAST NUCES · 2026*
