import asyncio
import base64
import os
import traceback

import anthropic
import httpx
from dotenv import load_dotenv
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

MODAL_VIDEO_URL = "https://ykzou1214--hy-worldplay-simple-generate-video-api.modal.run"
MODAL_HEALTH_URL = "https://ykzou1214--hy-worldplay-simple-health.modal.run"


# --- Models ---

class GenerateRequest(BaseModel):
    image_base64: str
    prompt: str = ""


class AnchorPoint(BaseModel):
    label: str
    x: float
    y: float


class SceneInfo(BaseModel):
    description: str
    mood: str
    anchor_points: list[AnchorPoint]
    sound_keywords: list[str]


class GenerateResponse(BaseModel):
    video_base64: str
    audio_base64: str | None = None
    scene: SceneInfo


# --- Services ---

async def analyze_scene(image_base64: str, prompt: str) -> SceneInfo:
    """Use Claude to analyze the image and extract scene metadata."""
    client = anthropic.AsyncAnthropic()

    # Detect media type from base64 header or default to jpeg
    media_type = "image/jpeg"
    if image_base64.startswith("/9j/"):
        media_type = "image/jpeg"
    elif image_base64.startswith("iVBOR"):
        media_type = "image/png"

    response = await client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"""Analyze this image for an immersive memory experience app. The user said: "{prompt}"

Return a JSON object (no markdown, just raw JSON) with:
{{
  "description": "A vivid 1-2 sentence description of the scene",
  "mood": "2-3 mood/emotion words, comma-separated",
  "anchor_points": [
    {{"label": "notable object or area", "x": 0.0-1.0, "y": 0.0-1.0}},
    ...up to 4 anchor points
  ],
  "sound_keywords": ["keyword1", "keyword2", ...up to 5 ambient sound keywords for this scene]
}}

The anchor points should identify interesting objects/areas in the image with normalized coordinates. Sound keywords should describe ambient sounds that would exist in this environment (e.g. "ocean waves", "wind", "birds chirping", "crowd murmur").""",
                    },
                ],
            }
        ],
    )

    import json
    raw = response.content[0].text.strip()
    # Handle potential markdown wrapping
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(raw)
    return SceneInfo(**data)


async def generate_video(image_base64: str, prompt: str) -> str:
    """Call the Modal endpoint to generate a video from the image."""
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(
            MODAL_VIDEO_URL,
            json={
                "prompt": prompt or "A scene",
                "image_base64": image_base64,
                "num_frames": 125,
                "pose": "w-31",
                "num_inference_steps": 30,
            },
        )
        result = response.json()
        if "error" in result:
            raise RuntimeError(f"Video generation failed: {result['error']}")
        return result["video_base64"]


async def generate_soundscape(sound_keywords: list[str]) -> str | None:
    """Use ElevenLabs to generate ambient audio from scene keywords."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return None

    from elevenlabs.client import AsyncElevenLabs

    client = AsyncElevenLabs(api_key=api_key)

    sound_prompt = ", ".join(sound_keywords)

    try:
        audio_gen = await client.text_to_sound_effects.convert(
            text=f"Ambient soundscape: {sound_prompt}. Continuous, immersive, no music.",
            duration_seconds=15.0,
        )

        # Collect audio bytes from async generator
        audio_bytes = b""
        async for chunk in audio_gen:
            audio_bytes += chunk

        return base64.b64encode(audio_bytes).decode()
    except Exception:
        traceback.print_exc()
        return None


# --- Endpoints ---

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Main endpoint: image in -> video + audio + scene metadata out."""
    try:
        # Run scene analysis and video generation in parallel
        scene_task = asyncio.create_task(analyze_scene(req.image_base64, req.prompt))
        video_task = asyncio.create_task(generate_video(req.image_base64, req.prompt))

        # Wait for scene analysis first (it's fast ~2-3s)
        scene = await scene_task

        # Start soundscape generation using scene keywords (runs while video is still generating)
        audio_task = asyncio.create_task(generate_soundscape(scene.sound_keywords))

        # Wait for video and audio
        video_base64, audio_base64 = await asyncio.gather(video_task, audio_task)

        return GenerateResponse(
            video_base64=video_base64,
            audio_base64=audio_base64,
            scene=scene,
        )
    except Exception as e:
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/health")
async def health():
    """Health check — also pings Modal endpoint."""
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
