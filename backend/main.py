import asyncio
import base64
import io
import os
import traceback

import fal_client
import httpx
from openai import OpenAI
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

# OpenAI client
openai_key = os.environ.get("OPENAI_API_KEY", "")
print(f"[startup] OPENAI_KEY loaded: {'yes' if openai_key else 'NO - MISSING'}")
openai_client = OpenAI(api_key=openai_key)

# ElevenLabs API key
elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "")
print(f"[startup] ELEVENLABS_KEY loaded: {'yes' if elevenlabs_key else 'NO - MISSING'}")

# Gradium API (STT + TTS)
gradium_key = os.environ.get("GRADIUM_API_KEY", "")
print(f"[startup] GRADIUM_KEY loaded: {'yes' if gradium_key else 'NO - MISSING'}")

GRADIUM_TTS_URL = "https://eu.api.gradium.ai/api/post/speech/tts"
GRADIUM_STT_URL = "https://eu.api.gradium.ai/api/post/speech/asr"
GRADIUM_VOICE_ID = "YTpq7expH9539ERJ"  # Emma voice

# Modal lingbot-world endpoints
MODAL_VIDEO_URL = "https://ykzou1214--lingbot-world-generate-video-api.modal.run"
MODAL_HEALTH_URL = "https://ykzou1214--lingbot-world-health.modal.run"

# Companion system prompt
COMPANION_SYSTEM_PROMPT = """You are a gentle, warm AI companion helping someone revisit a memory. You are present with them inside the memory world they have entered.

Context about the scene they are in: {scene_context}

Your role:
- Speak in a calm, warm, non-judgmental tone, like a trusted friend
- Reference specific details from the visual scene to ground the conversation
- Ask open-ended reflective questions that help them reconnect with the emotions and sensations of this memory
- Keep responses to 2-3 sentences maximum
- Never analyze or therapize — just be present and curious
- Use sensory language: "What does the air feel like here?" or "I notice the light..."
- If they share something emotional, acknowledge it gently before asking more"""


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


class SoundRequest(BaseModel):
    image_base64: str


class SoundResponse(BaseModel):
    audio_base64: str
    sound_prompt: str


class CompanionGreetRequest(BaseModel):
    scene_context: str


class CompanionGreetResponse(BaseModel):
    audio_base64: str
    text: str


class ConversationMessage(BaseModel):
    role: str
    content: str


class CompanionChatRequest(BaseModel):
    audio_base64: str
    conversation_history: list[ConversationMessage] = []
    scene_context: str


class CompanionChatResponse(BaseModel):
    audio_base64: str
    text: str
    user_text: str


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


async def _gradium_tts(text: str) -> str:
    """Convert text to speech via Gradium TTS. Returns base64-encoded WAV."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            GRADIUM_TTS_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": gradium_key,
            },
            json={
                "text": text,
                "voice_id": GRADIUM_VOICE_ID,
                "output_format": "wav",
                "only_audio": True,
            },
        )
        resp.raise_for_status()
        audio_bytes = resp.content
        print(f"[companion-tts] Got audio: {len(audio_bytes)} bytes")
        return base64.b64encode(audio_bytes).decode()


async def _gradium_stt(audio_base64: str) -> str:
    """Transcribe audio via Gradium STT. Returns text transcript."""
    audio_bytes = base64.b64decode(audio_base64)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            GRADIUM_STT_URL,
            headers={"x-api-key": gradium_key},
            files={"file": ("recording.webm", audio_bytes, "audio/webm")},
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"[companion-stt] Result: {result}")
        transcript = result.get("text", "") or result.get("transcript", "")
        return transcript.strip()


# --- Endpoints ---

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Proxy image to Modal AR endpoint, return generated video."""
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


@app.post("/api/generate-sound", response_model=SoundResponse)
async def generate_sound(req: SoundRequest):
    """Analyze image with OpenAI, then generate a looping sound effect via ElevenLabs."""
    try:
        # Step 1: Analyze image with OpenAI GPT-4o-mini
        print("[sound] Analyzing image with OpenAI...")
        vision_resp = await asyncio.to_thread(
            lambda: openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a sound designer. Analyze the image and describe "
                            "the ambient sound effects that would match this scene in "
                            "1-2 sentences. Focus on environmental and atmospheric sounds. "
                            "Be specific and vivid. Only output the sound description, nothing else."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{req.image_base64}",
                                },
                            },
                        ],
                    },
                ],
                max_tokens=150,
            )
        )
        sound_prompt = vision_resp.choices[0].message.content.strip()
        print(f"[sound] OpenAI sound prompt: {sound_prompt}")

        # Step 2: Generate sound effect with ElevenLabs
        print("[sound] Generating sound with ElevenLabs...")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.elevenlabs.io/v1/sound-generation",
                headers={
                    "Content-Type": "application/json",
                    "xi-api-key": elevenlabs_key,
                },
                json={
                    "text": sound_prompt,
                    "loop": True,
                    "duration_seconds": 10,
                    "model_id": "eleven_text_to_sound_v2",
                },
            )
            resp.raise_for_status()
            audio_bytes = resp.content
            print(f"[sound] Got audio: {len(audio_bytes)} bytes")

        audio_b64 = base64.b64encode(audio_bytes).decode()
        return SoundResponse(audio_base64=audio_b64, sound_prompt=sound_prompt)

    except Exception as e:
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/companion/greet", response_model=CompanionGreetResponse)
async def companion_greet(req: CompanionGreetRequest):
    """Generate an initial warm greeting referencing the user's memory scene."""
    try:
        print(f"[companion-greet] Generating greeting for scene: {req.scene_context[:80]}...")
        system_prompt = COMPANION_SYSTEM_PROMPT.format(scene_context=req.scene_context)

        chat_resp = await asyncio.to_thread(
            lambda: openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "The user has just entered this memory world. "
                            "Speak first — greet them warmly like an old friend. "
                            "Something like 'What brings you back here...' or 'How are you doing...' "
                            "Reference what you see in the scene. Keep it 2-3 sentences, gentle and curious."
                        ),
                    },
                ],
                max_tokens=150,
                temperature=0.8,
            )
        )
        greeting_text = chat_resp.choices[0].message.content.strip()
        print(f"[companion-greet] Greeting: {greeting_text}")

        audio_b64 = await _gradium_tts(greeting_text)
        return CompanionGreetResponse(audio_base64=audio_b64, text=greeting_text)

    except Exception as e:
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/companion/chat", response_model=CompanionChatResponse)
async def companion_chat(req: CompanionChatRequest):
    """Full voice conversation turn: STT -> LLM -> TTS."""
    try:
        # Step 1: Transcribe user speech via Gradium STT
        print("[companion-chat] Transcribing user speech...")
        user_text = await _gradium_stt(req.audio_base64)
        print(f"[companion-chat] User said: {user_text}")

        if not user_text:
            user_text = "(silence)"

        # Step 2: Build conversation for OpenAI
        system_prompt = COMPANION_SYSTEM_PROMPT.format(scene_context=req.scene_context)
        messages = [{"role": "system", "content": system_prompt}]
        for msg in req.conversation_history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_text})

        # Step 3: Get companion response
        print("[companion-chat] Getting LLM response...")
        chat_resp = await asyncio.to_thread(
            lambda: openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=150,
                temperature=0.8,
            )
        )
        companion_text = chat_resp.choices[0].message.content.strip()
        print(f"[companion-chat] Companion says: {companion_text}")

        # Step 4: Convert to speech via Gradium TTS
        audio_b64 = await _gradium_tts(companion_text)

        return CompanionChatResponse(
            audio_base64=audio_b64,
            text=companion_text,
            user_text=user_text,
        )

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
