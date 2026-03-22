"""Luxe FastAPI backend — adversarial image protection service."""

import asyncio
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

import config
from protect import Protector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
_protector: Protector | None = None
_processing_lock: asyncio.Lock | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _protector, _processing_lock

    # Create lock inside the running event loop (not get_event_loop — deprecated in 3.10+)
    _processing_lock = asyncio.Lock()

    logger.info("Initialising Protector (SegFormer)...")
    _protector = Protector()

    status = _protector.checkpoint_status()
    logger.info(f"Checkpoint status: {status}")
    if not status.get("cloak_unet"):
        logger.warning(
            "cloak_unet.pth MISSING — demo will fall back to full PGD (~90s/image). "
            "This must be present for demo day."
        )

    yield

    logger.info("Shutting down.")


app = FastAPI(title="Luxe Protection API", lifespan=lifespan)

# CORS — Vercel origin in prod, wildcard in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to Vercel URL in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Checkpoint-Status", "X-Processing-Path"],
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _validate_image(data: bytes) -> None:
    """Raise HTTPException for invalid or oversized images."""
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 10 MB)")

    # Two-buffer pattern: verify() destroys the handle, so we need a fresh BytesIO
    buf = BytesIO(data)
    try:
        Image.open(buf).verify()
    except Exception:
        raise HTTPException(400, "Invalid or corrupted image file")

    buf = BytesIO(data)
    try:
        Image.open(buf).convert("RGB")
    except Exception:
        # verify() passes silently for malformed WEBP in many Pillow versions;
        # convert("RGB") catches those edge cases.
        raise HTTPException(400, "Invalid or corrupted image file")


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    status = _protector.checkpoint_status() if _protector else {}
    return {"status": "ok", "checkpoints": status}


@app.post("/protect")
async def protect_endpoint(
    file: UploadFile = File(...),
    mode: Literal["full", "nudify", "modify"] = Form(...),
    texture: bool = Form(False),
):
    # Content-type validation
    ct = (file.content_type or "").lower()
    if not any(ct.startswith(t) for t in ("image/jpeg", "image/png", "image/webp")):
        raise HTTPException(415, "Unsupported media type — send JPEG, PNG, or WEBP")

    data = await file.read()
    _validate_image(data)

    if _processing_lock is None or _protector is None:
        raise HTTPException(503, "Server not ready")

    if _processing_lock.locked():
        raise HTTPException(429, "Processing in progress, try again shortly")

    async with _processing_lock:
        loop = asyncio.get_running_loop()
        png_bytes, checkpoint_status = await loop.run_in_executor(
            None, _protector.protect, data, mode, texture
        )

    processing_path = "unet" if checkpoint_status == "UNET_OK" else "pgd"

    return StreamingResponse(
        BytesIO(png_bytes),
        media_type="image/png",
        headers={
            "X-Checkpoint-Status": checkpoint_status,
            "X-Processing-Path": processing_path,
        },
    )
