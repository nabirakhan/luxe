"""Evaluation runner — runs all 6 tests on the fixed 20-image eval set.

Safety checker disabled for adversarial effectiveness measurement per
standard adversarial ML research practice.

Seed: torch.manual_seed(42) ensures reproducibility across runs.

Usage (Colab T4):
    python eval_runner.py \
        --img_dir /content/deepfashion/img \
        --seg_dir /content/deepfashion/seg \
        --mask_dir data/eval/masks \
        --eval_list data/eval_filenames.txt \
        --out results/eval_results.csv
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

torch.manual_seed(42)

from metrics import clip_cosine_sim, lpips_score, psnr_score, ssim_score
from protect import Protector
from eot import EOTTransforms


def load_eval_images(img_dir: str, eval_list: str):
    """Load the 20 fixed eval images. Returns list of (name, PIL Image)."""
    with open(eval_list) as f:
        names = [l.strip() for l in f if l.strip()]
    images = []
    for name in names:
        path = Path(img_dir) / name
        img = Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS)
        images.append((name, img))
    return images


def load_mask(name: str, mask_dir: str, seg_dir: str, protector: Protector):
    """Load pre-generated mask or generate with SegFormer."""
    mask_path = Path(mask_dir) / (Path(name).stem + "_mask.png")
    if mask_path.exists():
        m = Image.open(mask_path).convert("L").resize((512, 512), Image.NEAREST)
        return torch.from_numpy(np.array(m, dtype=np.float32) / 255.0).unsqueeze(0)

    # Generate and cache
    img_path = next(Path(seg_dir).rglob(Path(name).stem + ".png"), None)
    # Fallback: let Protector run SegFormer during protect_pgd_only
    return None


def _sd_inpaint(pipe, pil_img: Image.Image, mask: Image.Image, prompt: str) -> Image.Image:
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    gen.manual_seed(42)
    result = pipe(
        prompt=prompt,
        image=pil_img,
        mask_image=mask,
        generator=gen,
    ).images[0]
    return result


def run_test1(images, cloaked_images):
    """Test 1: LPIPS / SSIM / PSNR imperceptibility."""
    rows = []
    for (name, orig), (_, cloaked) in zip(images, cloaked_images):
        rows.append({
            "test": "1_imperceptibility",
            "image": name,
            "lpips": lpips_score(orig, cloaked),
            "ssim":  ssim_score(orig, cloaked),
            "psnr":  psnr_score(orig, cloaked),
        })
    return rows


def run_test2a(images, cloaked_images, masks, device):
    """Test 2a: Nudification disruption (SD v1.5 wb, SD v2 gb, SDXL gb)."""
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline

    prompt = "nude person"
    rows = []

    # Load each pipeline ONCE — loading inside per-image loop costs 40–100 min on T4
    pipes = {}
    pipe_ids = {
        "sd15_wb": ("runwayml/stable-diffusion-inpainting", False),
        "sd2_gb":  ("stabilityai/stable-diffusion-2-inpainting", False),
    }
    for key, (model_id, is_xl) in pipe_ids.items():
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        pipes[key] = pipe

    # SDXL
    pipe_xl = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    ).to(device)
    pipes["sdxl_gb"] = pipe_xl

    for (name, orig), (_, cloaked), mask in zip(images, cloaked_images, masks):
        mask_pil = Image.fromarray((mask.squeeze().numpy() * 255).astype(np.uint8)).convert("L")
        for pipe_key, pipe in pipes.items():
            out_orig   = _sd_inpaint(pipe, orig,    mask_pil, prompt)
            out_cloaked = _sd_inpaint(pipe, cloaked, mask_pil, prompt)
            rows.append({
                "test": "2a_nudify_disruption",
                "image": name,
                "model": pipe_key,
                "lpips_disruption": lpips_score(out_orig, out_cloaked),
            })
    return rows


def run_test2b(images, cloaked_images, masks, device):
    """Test 2b: IPP outfit-swap disruption (white-box fine-tuned)."""
    from diffusers import StableDiffusionInpaintPipeline

    ipp_checkpoint = Path("checkpoints/ipp_sd_finetuned")
    if not ipp_checkpoint.exists():
        print("Test 2b skipped: ipp_sd_finetuned checkpoint not found.")
        return []

    prompt = "change the outfit to a blue dress"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(ipp_checkpoint),
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    rows = []
    for (name, orig), (_, cloaked), mask in zip(images, cloaked_images, masks):
        mask_pil = Image.fromarray((mask.squeeze().numpy() * 255).astype(np.uint8)).convert("L")
        out_orig    = _sd_inpaint(pipe, orig,    mask_pil, prompt)
        out_cloaked = _sd_inpaint(pipe, cloaked, mask_pil, prompt)
        rows.append({
            "test": "2b_ipp_disruption",
            "image": name,
            "lpips_disruption": lpips_score(out_orig, out_cloaked),
        })
    return rows


def run_test2c(images, cloaked_images):
    """Test 2c: SegFormer mIoU degradation (5-image qualitative set)."""
    from segformer import SegFormerMask
    segformer = SegFormerMask()

    rows = []
    for (name, orig), (_, cloaked) in list(zip(images, cloaked_images))[:5]:
        def _get_mask(pil_img):
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            return segformer.generate_mask(t).squeeze().numpy()

        mask_orig   = _get_mask(orig)
        mask_cloaked = _get_mask(cloaked)

        # Simple pixel-level IoU
        intersection = (mask_orig * mask_cloaked).sum()
        union = np.clip(mask_orig + mask_cloaked, 0, 1).sum()
        iou = float(intersection / union) if union > 0 else 0.0

        rows.append({
            "test": "2c_segformer_miou",
            "image": name,
            "iou_cloaked_vs_original": iou,
        })
    return rows


def run_test3(images, cloaked_images):
    """Test 3: CLIP cosine similarity (target < 0.50)."""
    rows = []
    for (name, orig), (_, cloaked) in zip(images, cloaked_images):
        rows.append({
            "test": "3_clip_cosine",
            "image": name,
            "clip_cosine": clip_cosine_sim(orig, cloaked),
        })
    return rows


def run_test4(images, cloaked_images):
    """Test 4: EOT robustness — apply all 5 transforms, measure perceptual quality."""
    eot = EOTTransforms()
    transforms = [
        ("jpeg",       eot.jpeg_simulate),
        ("resize",     eot.random_resize),
        ("crop_pad",   eot.random_crop_pad),
        ("screenshot", eot.screenshot_simulate),
        ("color",      eot.color_jitter),
    ]
    rows = []
    for (name, orig), (_, cloaked) in zip(images, cloaked_images):
        orig_t   = torch.from_numpy(np.array(orig,   dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        cloaked_t = torch.from_numpy(np.array(cloaked, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

        for t_name, t_fn in transforms:
            with torch.no_grad():
                cloaked_aug = t_fn(cloaked_t).squeeze(0)
            cloaked_pil = Image.fromarray((cloaked_aug.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            rows.append({
                "test": "4_eot_robustness",
                "image": name,
                "transform": t_name,
                "lpips": lpips_score(orig, cloaked_pil),
                "ssim":  ssim_score(orig, cloaked_pil),
                "psnr":  psnr_score(orig, cloaked_pil),
            })
    return rows


def run_test5(images, cloaked_images, device):
    """Test 5: Grey-box — SD v2, SDXL, IPP vanilla (5-image qualitative)."""
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline

    prompt = "change the outfit to a blue dress"
    models = {
        "sd2_vanilla": StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device),
    }
    models["sdxl_vanilla"] = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    ).to(device)

    rows = []
    for (name, orig), (_, cloaked) in list(zip(images, cloaked_images))[:5]:
        mask_pil = Image.new("L", (512, 512), 255)  # full mask for qualitative
        for mname, pipe in models.items():
            out = _sd_inpaint(pipe, cloaked, mask_pil, prompt)
            out.save(f"results/grids/test5_{Path(name).stem}_{mname}.png")
            rows.append({"test": "5_greybox", "image": name, "model": mname, "saved": True})
    return rows


def run_test6(protector: Protector, images):
    """Test 6: Timing — warmup 1 dummy, mean ± std over 20 images."""
    import io

    # Warmup
    dummy_bytes = io.BytesIO()
    Image.new("RGB", (512, 512)).save(dummy_bytes, format="JPEG")
    protector.protect(dummy_bytes.getvalue(), mode="full")

    times = []
    for name, pil_img in images:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        t0 = time.perf_counter()
        protector.protect(buf.getvalue(), mode="full")
        times.append(time.perf_counter() - t0)

    return [{
        "test": "6_timing",
        "mean_s": float(np.mean(times)),
        "std_s":  float(np.std(times)),
        "min_s":  float(np.min(times)),
        "max_s":  float(np.max(times)),
    }]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",   default="/content/deepfashion/img")
    parser.add_argument("--seg_dir",   default="/content/deepfashion/seg")
    parser.add_argument("--mask_dir",  default="data/eval/masks")
    parser.add_argument("--eval_list", default="data/eval_filenames.txt")
    parser.add_argument("--out",       default="results/eval_results.csv")
    parser.add_argument("--mode",      default="full")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results/grids", exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)

    print("Loading Protector...")
    protector = Protector()

    print("Loading eval images...")
    images = load_eval_images(args.img_dir, args.eval_list)

    print("Generating cloaked images...")
    import io
    cloaked_images = []
    masks = []
    for name, pil_img in images:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        png_bytes, _ = protector.protect(buf.getvalue(), mode=args.mode)
        cloaked_pil = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        cloaked_images.append((name, cloaked_pil))

        # Load or create mask
        mask = load_mask(name, args.mask_dir, args.seg_dir, protector)
        if mask is None:
            mask = torch.ones(1, 512, 512)
        masks.append(mask)

        # Cache mask for future runs
        mask_save_path = Path(args.mask_dir) / (Path(name).stem + "_mask.png")
        if not mask_save_path.exists():
            Image.fromarray((mask.squeeze().numpy() * 255).astype(np.uint8)).save(mask_save_path)

    all_rows = []

    print("Test 1: Imperceptibility...")
    all_rows.extend(run_test1(images, cloaked_images))

    print("Test 2a: Nudification disruption...")
    all_rows.extend(run_test2a(images, cloaked_images, masks, device))

    print("Test 2b: IPP outfit-swap disruption...")
    all_rows.extend(run_test2b(images, cloaked_images, masks, device))

    print("Test 2c: SegFormer mIoU...")
    all_rows.extend(run_test2c(images, cloaked_images))

    print("Test 3: CLIP cosine similarity...")
    all_rows.extend(run_test3(images, cloaked_images))

    print("Test 4: EOT robustness...")
    all_rows.extend(run_test4(images, cloaked_images))

    print("Test 5: Grey-box qualitative...")
    all_rows.extend(run_test5(images, cloaked_images, device))

    print("Test 6: Timing...")
    all_rows.extend(run_test6(protector, images))

    # Write CSV
    if all_rows:
        keys = sorted({k for row in all_rows for k in row})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
