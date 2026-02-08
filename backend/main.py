import asyncio
import base64
import io
import os
import traceback

import fal_client
import httpx
from dotenv import load_dotenv
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Rewind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# fal.ai API key
fal_key = os.environ.get("FAL_AI_API_KEY", "")
print(f"[startup] FAL_KEY loaded: {'yes' if fal_key else 'NO - MISSING'}  (len={len(fal_key)})")
os.environ["FAL_KEY"] = fal_key

# Nebius VM endpoint (8x H200)
NEBIUS_BASE_URL = os.environ.get("NEBIUS_URL", "http://66.201.6.236:8888")
NEBIUS_VIDEO_URL = f"{NEBIUS_BASE_URL}/generate"
NEBIUS_HEALTH_URL = f"{NEBIUS_BASE_URL}/health"


# --- Models ---

class GenerateRequest(BaseModel):
    image_base64: str
    prompt: str = ""
    pose: str = "w-3"


class GenerateResponse(BaseModel):
    video_base64: str
    prompt: str
    num_frames: int
    pose: str
    generation_time_seconds: float


# --- Helpers ---

def _is_16_9(image_base64: str) -> bool:
    """Check if a base64-encoded image is already 16:9 (with small tolerance)."""
    img = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    w, h = img.size
    ratio = w / h
    return abs(ratio - 16 / 9) < 0.05


def _on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(f"[fal] {log['message']}")


def _convert_to_16_9(image_base64: str, prompt: str) -> str:
    """Use fal.ai nano-banana to extend an image to 16:9 aspect ratio.
    Returns the resulting image URL.
    """
    # Upload image to fal storage to get a real URL
    image_bytes = base64.b64decode(image_base64)
    uploaded_url = fal_client.upload(image_bytes, content_type="image/jpeg")
    print(f"[fal] Uploaded image to: {uploaded_url}")

    result = fal_client.subscribe(
        "fal-ai/nano-banana/edit",
        arguments={
            "prompt": prompt or "extend the image to 16:9",
            "num_images": 1,
            "aspect_ratio": "16:9",
            "output_format": "png",
            "image_urls": [uploaded_url],
        },
        with_logs=True,
        on_queue_update=_on_queue_update,
    )

    print(f"[fal] Result: {result}")
    image_url = result["images"][0]["url"]
    return image_url


# --- Endpoints ---

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Proxy image to Nebius endpoint, return generated video."""
    try:
        # Step 1: Skip 16:9 conversion for WASD navigation (captured frames are already 16:9)
        is_wasd = req.pose.split("-")[0] in ("w", "a", "s", "d")

        if is_wasd:
            print(f"[generate] WASD pose '{req.pose}' — skipping 16:9 check")
            image_16_9_b64 = req.image_base64
        else:
            print(f"[generate] Checking aspect ratio...")
            is_16_9 = _is_16_9(req.image_base64)
            print(f"[generate] Already 16:9: {is_16_9}")

            if is_16_9:
                image_16_9_b64 = req.image_base64
            else:
                print(f"[generate] Calling fal.ai nano-banana...")
                image_url = await asyncio.to_thread(
                    _convert_to_16_9, req.image_base64, req.prompt
                )
                print(f"[generate] Got image URL: {image_url[:100]}...")
                async with httpx.AsyncClient(timeout=120) as client:
                    img_resp = await client.get(image_url)
                    img_resp.raise_for_status()
                    image_16_9_b64 = base64.b64encode(img_resp.content).decode()
                print(f"[generate] Downloaded converted image, b64 len={len(image_16_9_b64)}")

        # Step 2: Send image to Nebius for video generation
        print(f"[generate] Sending to Nebius...")
        async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
            response = await client.post(
                NEBIUS_VIDEO_URL,
                json={
                    "prompt": req.prompt or "A scene",
                    "image_base64": image_16_9_b64,
                    "num_frames": 9,
                    "pose": req.pose,
                    "sampling_steps": 15,
                    "guide_scale": 3.0,
                    "max_area": "720*1280",
                },
            )
            print(f"[generate] Nebius response status: {response.status_code}")
            result = response.json()
            print(f"[generate] Nebius response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            if isinstance(result, dict) and "error" in result:
                print(f"[generate] Nebius error: {result['error']}")
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=500, content=result)
            return result
    except Exception as e:
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/health")
async def health():
    """Health check — pings Nebius endpoint."""
    nebius_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(NEBIUS_HEALTH_URL)
            nebius_status = resp.json()
    except Exception as e:
        nebius_status = {"error": str(e)}

    return {
        "status": "healthy",
        "nebius": nebius_status,
    }
