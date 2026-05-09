# Luxe — Adversarial Image Protection

**Your Photos. Your Rules.**

Luxe shields personal photos from AI-powered nudification and outfit-modification attacks using imperceptible adversarial perturbations. It jointly attacks both inpainting nudification pipelines (SD v1.5/v2/SDXL) and instruction-following modification models (InstructPix2Pix, IP-Adapter) within a unified PGD framework augmented by Expectation over Transformations (EOT) and a CLIP ensemble second pass.

Deep Learning for Perceptron — May 2026 · Nabira Khan · Rameen · Aisha

---

## Key Results

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Visual quality (LPIPS) | 0.077 | < 0.10 | ✅ PASS |
| Visual quality (PSNR) | 37.2 dB | > 35 dB | ✅ PASS |
| CLIP semantic disruption | 0.234 | < 0.50 | ✅ PASS |
| CloakUNet inference speed | **0.14s** | < 1.0s | ✅ PASS (630× faster than PGD) |
| SD v1.5 white-box disruption | 0.164 | — | — |
| SD v2 grey-box disruption | 0.177 | — | — |

---

## How It Works

```
Upload → centre-crop → resize 512×512
       → SegFormer-B2 clothing/skin mask
       → PGD inpainting attack (40 steps, ε=8/255, vs SD v1.5 VAE, EOT active)
       → PGD modification attack (40 steps, vs InstructPix2Pix + IP-Adapter, EOT active)
       → Delta merge (inpaint + 0.5×mod, re-clipped to ε=8/255)
       → CLIP ensemble attack (40 steps, ε=16/255, vs ViT-B/32 + ViT-L/14)
       → LPIPS safety gate (scale down if > 0.10)
       → 512×512 protected PNG
```

**Fast path (deployment):** CloakUNet v5 replaces the full PGD pipeline — single forward pass, 0.14s on T4 GPU, <10s on CPU.

---

## Models

| Checkpoint | Description |
|------------|-------------|
| `segformer_lip.pth` | SegFormer-B2 fine-tuned on LIP (20-class body-part segmentation) |
| `cloak_unet.pth` | CloakUNet v5 — amortised adversarial delta predictor |
| `sd_inpaint_vae.pth` | Fine-tuned SD v1.5 Inpainting VAE (nudification surrogate) |
| `ipp_vae.pth` | IPP VAE (modification surrogate) |
| `ip_adapter.pth` | IP-Adapter conditioning target (outfit-swap guard) |

Checkpoints are downloaded automatically at startup from Google Drive (see `backend/download_checkpoints.py`).

---

## Local Development

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Health check: `http://localhost:8000/health` — all 5 checkpoints should show `true`.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local   # edit VITE_API_URL if needed
npm run dev
```

---

## Deployment

### Backend → Render

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New → Web Service** → connect the repo, root `backend/`.
3. Set env vars in Render dashboard:
   ```
   CORS_ORIGIN = https://<your-vercel-url>.vercel.app
   ```
4. Checkpoints are downloaded automatically on first startup via `download_checkpoints.py`.
5. Verify: `/health` should return all 5 checkpoints `true`.

> **Plan**: Standard (2 GB RAM) minimum for SegFormer + UNet at startup.

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → **New Project** → import the GitHub repo.
2. Set **Root Directory** to `frontend`.
3. Add environment variable:
   ```
   VITE_API_URL = https://<your-render-service>.onrender.com
   ```
4. Deploy — Vercel detects Vite automatically.

---

## API

### `GET /health`
```json
{
  "status": "ok",
  "checkpoints": {
    "segformer_lip": true,
    "cloak_unet": true,
    "sd_inpaint_vae": true,
    "ipp_vae": true,
    "ip_adapter": true
  }
}
```

### `POST /protect`
| Field | Type | Values |
|-------|------|--------|
| `file` | image (JPEG/PNG/WEBP, max 10 MB) | — |
| `mode` | string | `nudify` · `modify` · `full` |
| `texture` | bool | `true` / `false` |

Response: `image/png` (512×512)  
Headers: `X-Checkpoint-Status` (`UNET_OK` or `UNET_MISSING`), `X-Processing-Path` (`unet` or `pgd`)

---

## System Architecture

| Layer | Technology | Hosting |
|-------|------------|---------|
| Frontend | React 18 + Vite + Three.js + Framer Motion + Tailwind | Vercel |
| Backend API | Python 3.10 + FastAPI + Uvicorn | Render |
| Adversarial engine | PyTorch 2.x, PGD, EOT | Backend |
| Segmentation | SegFormer-B2 (fine-tuned LIP) | Backend |
| Nudification surrogate | SD v1.5 Inpainting VAE (fine-tuned) | Backend |
| Modification surrogate | IP-Adapter + CLIP ViT-B/32 | Backend |
| Fast path | CloakUNet v5 | Backend |
| CLIP attack | ViT-B/32 + ViT-L/14 ensemble | Backend |
| Training compute | Kaggle T4 GPU (15.6 GB VRAM) | — |

---

## EOT Robustness

| Transform | Status |
|-----------|--------|
| Random Resize | ✅ GOOD (0.93× baseline) |
| Colour Jitter | ✅ GOOD (1.09× baseline) |
| JPEG Compression | ⚠️ PARTIAL (1.83× baseline) |
| Crop + Pad | ❌ DEGRADED (4.35× baseline) |
| Screenshot Sim | ❌ DEGRADED (5.74× baseline) |

---

## References

- Salman et al. (2023). PhotoGuard. arXiv:2302.06588.
- Athalye et al. (2018). EOT. ICML 2018.
- Shan et al. (2023). Glaze. USENIX Security 2023.
- Madry et al. (2018). PGD. ICLR 2018.
- Radford et al. (2021). CLIP. ICML 2021.
- Xie et al. (2021). SegFormer. NeurIPS 2021.
- Rombach et al. (2022). Stable Diffusion. CVPR 2022.
- Brooks et al. (2023). InstructPix2Pix. CVPR 2023.
- Ye et al. (2023). IP-Adapter. arXiv:2308.06721.
