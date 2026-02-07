# Rewind — Engineering Plan

## Team

| Person | Role | Scope |
|--------|------|-------|
| **Christian** | Backend | Python backend, API integration, scene analysis |
| **Yongkang** | Backend | Python backend, model connection, Modal deployment |
| **Alessandro** | Frontend | Web UI, 3D viewer, user experience |

---

## Backend (Christian + Yongkang)

**Stack:** Python (FastAPI)

### What you're building

A Python API server that sits between the frontend and all AI services. The frontend sends a photo — the backend orchestrates everything and returns the results.

### Endpoints to build

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Main endpoint. Accepts an image, kicks off video generation + soundscape + scene analysis. Returns video + audio + metadata. |
| `/api/health` | GET | Backend health + status of downstream services |

### Flow for `/api/generate`

```
Frontend sends image (base64) + prompt
        │
        ▼
Backend receives request
        │
        ├─── 1. Scene Analysis (parallel)
        │    - Use an LLM (e.g. GPT-4o / Claude) to analyze the image
        │    - Extract: environment type, mood, objects, suggested sounds
        │    - Return: scene description, sound keywords, anchor points
        │
        ├─── 2. Video Generation (parallel)
        │    - POST to Modal endpoint (HY-WorldPlay)
        │    - https://ykzou1214--hy-worldplay-simple-generate-video-api.modal.run
        │    - Send: image_base64, prompt, pose, num_frames
        │    - Receive: video_base64 (MP4)
        │
        └─── 3. Soundscape Generation (parallel, after scene analysis)
             - Use ElevenLabs API to generate ambient audio
             - Based on scene analysis keywords (ocean, wind, crowd, etc.)
             - Receive: audio file(s)
        │
        ▼
Backend returns to frontend:
{
  "video_base64": "...",
  "audio_url": "...",
  "scene": { "description": "...", "mood": "...", "anchor_points": [...] },
}
```

### Task split suggestion

**Yongkang:**
- Modal endpoint connection (already done — see `modal-deployment.md`)
- `/api/generate` endpoint skeleton + Modal video generation call
- Wire up the full orchestration (parallel calls, combine results)

**Christian:**
- Scene analysis service (LLM call to analyze image → extract environment, mood, objects, sounds)
- ElevenLabs soundscape integration (generate ambient audio from scene keywords)
- AI companion voice conversation endpoint (if time permits)

### Key decisions

- **Run scene analysis + video generation in parallel** — scene analysis is fast (~2-3s), video gen is slow (~2-5 min). Start both at once, then use scene analysis output to generate soundscape while video is still rendering.
- **Keep it simple** — single FastAPI app, no database, no auth. Everything stateless.
- **Return base64** — video and audio returned as base64 in JSON response. No file storage needed for the hackathon.

### Expected deliverables

- [ ] FastAPI server with `/api/generate` endpoint
- [ ] Modal video generation integration (image → video)
- [ ] Scene analysis (image → description, mood, sound keywords, anchor points)
- [ ] ElevenLabs soundscape generation (keywords → ambient audio)
- [ ] AI companion conversation endpoint (stretch goal)

---

## Frontend (Alessandro)

**Stack:** Web (framework TBD — React / Next.js / vanilla)

### What you're building

A web app where users upload a photo, wait for generation, then experience an immersive memory world with video playback, spatial audio, and optional AI companion interaction.

### Screens / Views

1. **Upload Screen**
   - Photo picker / drag-and-drop
   - Optional text input ("wish I could go back to that time")
   - "Enter Memory" button → calls backend `/api/generate`

2. **Loading Screen**
   - "Entering your memory..." transition
   - Show the original photo with a subtle animation
   - Play light ambient audio preview while waiting (~2-5 min wait)

3. **Memory World View**
   - Full-screen video playback of the generated world
   - Ambient soundscape playing over the video
   - Anchor point overlays (clickable hotspots on the video)
   - Click anchor point → triggers AI companion voice conversation

4. **Companion Conversation (stretch)**
   - Voice-based chat overlay
   - Calm companion asks reflective questions about the memory
   - 2-5 exchange conversation per anchor point

### What the backend gives you

```json
POST /api/generate
Request:  { "image_base64": "...", "prompt": "..." }
Response: {
  "video_base64": "<MP4 as base64>",
  "audio_url": "<ambient audio>",
  "scene": {
    "description": "A beach at sunset with gentle waves",
    "mood": "peaceful, nostalgic",
    "anchor_points": [
      { "label": "Ocean waves", "x": 0.3, "y": 0.6 },
      { "label": "Beach chair", "x": 0.7, "y": 0.5 }
    ]
  }
}
```

### Expected deliverables

- [ ] Upload screen with image picker
- [ ] Loading/transition screen
- [ ] Memory world viewer (video + audio playback)
- [ ] Anchor point display on video (stretch)
- [ ] AI companion conversation UI (stretch)

---

## Timeline (Hackathon)

| Phase | What | Who |
|-------|------|-----|
| **Hour 1-2** | Backend: FastAPI skeleton + Modal integration. Frontend: Upload screen + loading screen. | All |
| **Hour 3-4** | Backend: Scene analysis + ElevenLabs. Frontend: Memory world viewer. | All |
| **Hour 5-6** | Integration: Frontend ↔ Backend connected end-to-end. Polish. | All |
| **Hour 7+** | Stretch: AI companion, anchor points, polish UX. | All |

## API Contract Summary

This is the single contract between frontend and backend:

```
POST /api/generate
Content-Type: application/json

Request:
{
  "image_base64": "<base64 encoded image>",
  "prompt": "optional description"
}

Response (success):
{
  "video_base64": "<base64 MP4>",
  "audio_url": "<base64 or URL to ambient audio>",
  "scene": {
    "description": "string",
    "mood": "string",
    "anchor_points": [
      { "label": "string", "x": 0.0-1.0, "y": 0.0-1.0 }
    ]
  }
}

Response (error):
{
  "error": "string"
}
```
