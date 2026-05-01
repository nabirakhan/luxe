# Luxe — Adversarial Image Protection

Luxe shields personal photos from AI-powered abuse tools — nudification pipelines, outfit-swap models, and appearance-modification networks — using imperceptible adversarial perturbations.

## Training Status

| Component | Status |
|-----------|--------|
| SegFormer (LIP, 20-class) | ✅ Complete |
| SD Inpaint VAE fine-tune | ✅ Complete |
| IPP VAE | ✅ Complete |
| IP-Adapter | ✅ Complete |
| CloakUNet (fast amortised attacker) | ✅ Complete |

All five checkpoints present. Fast-path inference: **~3.7 s/image** on CPU.

---

## Architecture

```
Upload → SegFormer mask → CloakUNet (fast path, <10s)
                       ↘ PGD + CLIP fallback (~90s, if UNet missing)
       → 512×512 protected PNG
```

**Models used:**
- `segformer_lip.pth` — 20-class LIP segmentation, isolates clothing + skin regions
- `cloak_unet.pth` — amortised adversarial delta predictor (encoder-decoder, skip connections)
- `sd_inpaint_vae.pth` — surrogate VAE for inpaint-nudification loss
- `ipp_vae.pth` — surrogate VAE for IPP modification loss
- `ip_adapter.pth` — IP-Adapter conditioning target for outfit-swap guard

---

## Local Development

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Health check: `http://localhost:8000/health` — all 5 checkpoints should show `true`.

Texture calibration (run once after training, needs deepfashion images):
```bash
py texture.py --calibrate --img_dir /path/to/deepfashion/img --seg_dir /path/to/deepfashion/seg
```
Confirm `LPIPS < 0.10` for all amplitudes before demo.

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
2. Go to [render.com](https://render.com) → **New → Blueprint** → connect the repo.  
   Render reads `render.yaml` automatically.
3. After the first deploy, go to **Disks** for the `luxe-backend` service.  
   Upload the checkpoint files to the mounted disk:
   ```
   /opt/render/project/src/backend/checkpoints/segformer_lip.pth   (105 MB)
   /opt/render/project/src/backend/checkpoints/cloak_unet.pth      (15 MB)
   /opt/render/project/src/backend/checkpoints/sd_inpaint_vae.pth  (320 MB)
   /opt/render/project/src/backend/checkpoints/ipp_vae.pth         (320 MB)
   /opt/render/project/src/backend/checkpoints/ip_adapter.pth      (1.6 MB)
   ```
   Use Render's shell (`Connect → Shell`) to pull files from Google Drive or copy via `scp`.
4. Set `CORS_ORIGIN` env var in Render dashboard to your Vercel URL (e.g. `https://luxe.vercel.app`).
5. Trigger a manual redeploy — `/health` should return all 5 checkpoints `true`.

> **Plan**: Standard (2 GB RAM) is the minimum for SegFormer + UNet at startup.

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → **New Project** → import the GitHub repo.
2. Set **Root Directory** to `frontend`.
3. Add environment variable:
   ```
   VITE_API_URL = https://luxe-backend.onrender.com
   ```
   (replace with your actual Render service URL)
4. Deploy. Vercel detects Vite automatically (`vercel.json` is also present as a fallback).

### Post-deploy CORS

After Vercel gives you a URL (e.g. `https://luxe-abc123.vercel.app`):

1. In Render dashboard → `luxe-backend` → **Environment** → set:
   ```
   CORS_ORIGIN = https://luxe-abc123.vercel.app
   ```
2. Redeploy the Render service.

### Smoke test

```bash
# Health
curl https://luxe-backend.onrender.com/health

# Fast path (should return in <10s once warmed up)
curl -s -o out.png -w "%{http_code} %{time_total}s\n" \
  -F "file=@test.jpeg;type=image/jpeg" \
  -F "mode=full" -F "texture=false" \
  https://luxe-backend.onrender.com/protect
```

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
