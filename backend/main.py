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

# Modal lingbot-world endpoints
MODAL_VIDEO_URL = "https://ykzou1214--lingbot-world-generate-video-api.modal.run"
MODAL_HEALTH_URL = "https://ykzou1214--lingbot-world-health.modal.run"


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


def _convert_to_16_9(image_base64: str, prompt: str) -> str:
    """Use fal.ai nano-banana to extend an image to 16:9 aspect ratio.
    Returns the resulting image URL.
    """
    data_uri = f"data:image/png;base64,{image_base64}"

    result = fal_client.subscribe(
        "fal-ai/nano-banana/edit",
        arguments={
            "prompt": prompt or "extend the image to 16:9",
            "num_images": 1,
            "aspect_ratio": "16:9",
            "output_format": "png",
            "image_urls": [data_uri],
        },
        with_logs=True,
        on_queue_update=lambda update: None,
    )

    image_url = result["images"][0]["url"]
    return image_url


# --- Endpoints ---

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Proxy image to Modal AR endpoint, return generated video."""
    try:
        # Step 1: Only convert to 16:9 if the image isn't already 16:9
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

        # Step 2: Send 16:9 image to Modal lingbot-world for video generation
        print(f"[generate] Sending to Modal...")
        async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
            response = await client.post(
                MODAL_VIDEO_URL,
                json={
                    "prompt": req.prompt or "A scene",
                    "image_base64": image_16_9_b64,
                    "num_frames": 17,
                    "pose": req.pose,
                    "sampling_steps": 40,
                    "max_area": "720*1280",
                },
            )
            print(f"[generate] Modal response status: {response.status_code}")
            result = response.json()
            print(f"[generate] Modal response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            if isinstance(result, dict) and "error" in result:
                print(f"[generate] Modal error: {result['error']}")
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=500, content=result)
            return result
    except Exception as e:
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/health")
async def health():
    """Health check — pings Modal endpoint."""
    modal_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(MODAL_HEALTH_URL)
            modal_status = resp.json()
    except Exception as e:
        modal_status = {"error": str(e)}

    return {
        "status": "healthy",
        "modal": modal_status,
    }
