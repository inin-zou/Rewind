# Rewind — Backend Architecture

## Overview

The backend orchestrates a **streaming chunk-based world exploration** experience. A user uploads a photo, the system generates short video chunks (13 frames each) showing camera movement through the scene, and streams them to the frontend in real-time. Each chunk's last frame seeds the next generation, creating an unlimited explorable experience.

The key insight from WorldPlay's design: **don't generate long videos — generate one chunk at a time and stream.**

---

## Performance Benchmarks (2x A100-80GB, torch.compile + SageAttention)

| Frames | Latents | Duration (24fps) | Inference Time | Notes |
|--------|---------|-------------------|----------------|-------|
| 13 | 4 (1 chunk) | 0.54s | **4.2s** | Streaming target |
| 61 | 16 (4 chunks) | 2.5s | **23.0s** | Medium batch |
| 125 | 32 (8 chunks) | 5.2s | 67-71s | Full batch (old approach) |

First request on a fresh container has ~110s torch.compile warmup. After that, compiled cache is hot and times above apply.

---

## Why Chunked Streaming Works (WorldPlay Design)

WorldPlay achieves low latency through three mechanisms:

1. **Next-chunk prediction (13 frames):** Instead of generating a full 5s video, the model predicts only the next short chunk. This keeps compute cost fixed and TTFV (time-to-first-video) low.

2. **Few-step distillation (4 steps):** The AR distilled model compresses diffusion sampling from 30 to 4 steps using Context Forcing, matching the bidirectional teacher's quality.

3. **Memory/context forcing:** Maintains long-range consistency across chunks despite few-step generation, preventing drift that would otherwise compound over many chunks.

**For our product:** TTFV is ~4.2s. The user sees the first chunk within 4 seconds of pressing a direction, then every ~4 seconds they get another 0.54s of video.

---

## User Flow

```
1. User uploads a photo
2. Backend runs scene analysis (Claude) + first video chunk (Modal) + soundscape (ElevenLabs) in parallel
3. User sees the first 0.54s chunk + ambient audio within ~5s
4. User holds a direction (WASD) or moves joystick
5. Backend streams chunks continuously:
   - Extract last frame from previous chunk → input for next chunk
   - Generate next 13-frame chunk with user's current pose
   - Push chunk to frontend via WebSocket/SSE
   - Repeat while user holds direction
6. User releases input → generation stops
7. User presses new direction → new stream of chunks begins
```

---

## Streaming Protocol

### WebSocket: `/ws/memories/{memory_id}/explore`

Bidirectional WebSocket for real-time chunk streaming.

**Client → Server messages:**

```json
// Start/continue generating chunks in a direction
{ "action": "move", "pose": "w-3" }

// Stop generating
{ "action": "stop" }

// Change direction mid-stream
{ "action": "move", "pose": "a-3, right-1" }
```

**Server → Client messages:**

```json
// Chunk ready
{
  "type": "chunk",
  "chunk_id": "uuid",
  "chunk_index": 5,
  "video_base64": "<base64 MP4, 13 frames>",
  "generation_time_seconds": 4.2,
  "pose": "w-3"
}

// Generation status
{ "type": "status", "message": "generating", "chunk_index": 6 }

// Error
{ "type": "error", "message": "Worker crashed" }
```

**Frontend playback strategy:**
- Buffer chunks in a queue
- Play each chunk (0.54s) seamlessly after the previous one
- Request next chunk as soon as current one starts playing (pipeline ahead)
- With ~4s generation per 0.54s of video, there's an ~8:1 ratio — user watches 0.54s, waits ~3.5s for next

### Fallback: SSE `/api/memories/{memory_id}/stream`

For clients that can't use WebSocket (e.g. mobile browsers):

```
POST /api/memories/{memory_id}/stream
Content-Type: application/json

{ "pose": "w-3", "num_chunks": 5 }
```

Returns Server-Sent Events:

```
event: chunk
data: {"chunk_id":"uuid","chunk_index":0,"video_base64":"...","generation_time_seconds":4.2}

event: chunk
data: {"chunk_id":"uuid","chunk_index":1,"video_base64":"...","generation_time_seconds":4.1}

event: done
data: {"total_chunks":5,"total_time_seconds":21.5}
```

---

## REST API (Non-Streaming)

### `POST /api/memories` — Start a new memory

Creates a new memory session from an uploaded photo. Runs scene analysis, generates the first video chunk (13 frames), and produces the soundscape.

**Request:**
```json
{
  "image_base64": "<base64 encoded photo>",
  "prompt": "optional description of the memory"
}
```

**Response:**
```json
{
  "memory_id": "uuid",
  "chunk": {
    "chunk_id": "uuid",
    "video_base64": "<base64 MP4, 13 frames>",
    "pose": "w-3",
    "chunk_index": 0,
    "generation_time_seconds": 4.2
  },
  "audio_base64": "<base64 ambient audio>",
  "scene": {
    "description": "A beach at sunset with gentle waves",
    "mood": "peaceful, nostalgic",
    "anchor_points": [
      { "label": "Ocean waves", "x": 0.3, "y": 0.6 },
      { "label": "Beach chair", "x": 0.7, "y": 0.5 }
    ],
    "sound_keywords": ["ocean waves", "seagulls", "wind"]
  }
}
```

### `POST /api/memories/{memory_id}/chunks` — Generate a single chunk

For non-streaming clients. Generates one 13-frame chunk.

**Request:**
```json
{
  "pose": "a-3, right-1",
  "parent_chunk_id": "uuid"
}
```

**Response:**
```json
{
  "chunk": {
    "chunk_id": "uuid",
    "video_base64": "<base64 MP4, 13 frames>",
    "pose": "a-3, right-1",
    "chunk_index": 3,
    "generation_time_seconds": 4.2
  }
}
```

### `GET /api/memories/{memory_id}` — Get memory details

Retrieve a memory and its full chunk tree for replaying a session.

### `GET /api/health` — Health check

```json
{
  "status": "healthy",
  "modal": { "status": "healthy", "pipeline_loaded": true, "workers_alive": true },
  "supabase": "connected"
}
```

---

## Pose Mapping (Frontend → Backend)

With 13-frame chunks (4 latent frames), the pose step count is **3** (latent_num - 1 = 4 - 1 = 3). The worker auto-adjusts pose duration to match the frame count.

| Frontend Input | Pose String | Description |
|----------------|-------------|-------------|
| W key | `w-3` | Walk forward (1 chunk) |
| S key | `s-3` | Walk backward |
| A key | `a-3` | Strafe left |
| D key | `d-3` | Strafe right |
| Joystick up | `up-N` | Pitch camera up |
| Joystick down | `down-N` | Pitch camera down |
| Joystick left | `left-N` | Yaw camera left |
| Joystick right | `right-N` | Yaw camera right |
| W + joystick right | `w-2, right-1` | Walk forward while turning right |

For longer chunks (e.g. 61 frames = 16 latents), step count would be 15. The backend auto-adjusts.

---

## Supabase Schema

### `memories` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | uuid (PK) | Memory ID |
| `user_id` | uuid (FK → auth.users) | Owner |
| `prompt` | text | User's description |
| `scene_description` | text | Claude's scene analysis |
| `scene_mood` | text | Mood keywords |
| `anchor_points` | jsonb | Array of `{label, x, y}` |
| `sound_keywords` | jsonb | Array of sound keyword strings |
| `original_image_path` | text | Path in Supabase Storage |
| `created_at` | timestamptz | Creation time |

### `chunks` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | uuid (PK) | Chunk ID |
| `memory_id` | uuid (FK → memories) | Parent memory |
| `parent_chunk_id` | uuid (FK → chunks, nullable) | Previous chunk (null for first) |
| `chunk_index` | int | Order in the exploration path |
| `pose` | text | Pose string used (e.g. `"w-3"`) |
| `num_frames` | int | Frame count (default 13) |
| `video_path` | text | Path in Supabase Storage |
| `last_frame_path` | text | Path in Supabase Storage (input for next chunk) |
| `generation_time` | float | Seconds to generate |
| `created_at` | timestamptz | Creation time |

### `sessions` table (optional, for replay)

| Column | Type | Description |
|--------|------|-------------|
| `id` | uuid (PK) | Session ID |
| `memory_id` | uuid (FK → memories) | Which memory |
| `user_id` | uuid (FK → auth.users) | Who explored |
| `chunk_path` | jsonb | Ordered array of chunk IDs representing the user's journey |
| `created_at` | timestamptz | Session start |

### Supabase Storage Buckets

| Bucket | Contents |
|--------|----------|
| `originals` | Uploaded photos |
| `videos` | Generated MP4 chunks (13 frames each) |
| `frames` | Extracted last frames (PNG) for seeding next chunk |
| `audio` | Generated soundscape files |

---

## Backend Internal Flow

### First Chunk (Memory Creation)

```
POST /api/memories { image_base64, prompt }
    │
    ├─── [parallel] Claude: analyze scene → SceneInfo
    ├─── [parallel] Modal: generate 13-frame chunk (pose="w-3") → video_base64
    │
    │  (wait for scene analysis)
    │
    ├─── [parallel] ElevenLabs: generate soundscape → audio_base64
    │
    │  (wait for all)
    │
    ├─── Extract last frame from video MP4
    ├─── Upload to Supabase Storage: original image, video, last frame, audio
    ├─── Insert into Supabase: memories row + first chunks row
    │
    └─── Return response (~5s total)
```

### Streaming Chunks (Exploration)

```
WebSocket /ws/memories/{id}/explore
    │
    │  Client: { "action": "move", "pose": "w-3" }
    │
    ├── Loop while user holds direction:
    │   │
    │   ├─── Fetch last frame from previous chunk (in-memory or Supabase)
    │   ├─── Modal: generate 13-frame chunk (image=last_frame, pose=pose)
    │   │    └── ~4.2s with warm torch.compile cache
    │   ├─── Extract last frame from new chunk
    │   ├─── Push chunk to client via WebSocket
    │   ├─── Store chunk + last frame (async, non-blocking)
    │   │
    │   └── Continue if client still sending "move"
    │
    │  Client: { "action": "stop" }
    │
    └── Stop generating, keep connection open
```

---

## Latency Optimization Stack

| Layer | Technique | Impact |
|-------|-----------|--------|
| Model | AR distilled (4 steps vs 30) | 7.5x fewer forward passes |
| Model | torch.compile | ~2x kernel fusion speedup |
| Model | SageAttention | Optimized attention kernels |
| Architecture | Persistent worker daemon | No model reload per request (~200s saved) |
| Architecture | No model offloading | No CPU↔GPU transfers (~2.7x speedup) |
| Architecture | 13-frame chunks (not 125) | 8x less compute per request |
| Infrastructure | `min_containers=1` | No cold start for first user |
| Infrastructure | File-based IPC (not subprocess) | Persistent torchrun, model stays in VRAM |

**Net result:** 250s → 4.2s per chunk (60x speedup)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| API server | FastAPI (Python) with WebSocket support |
| Package manager | uv |
| Video generation | HY-WorldPlay AR distilled on Modal (2x A100-80GB) |
| Scene analysis | Claude (Anthropic API) |
| Soundscape | ElevenLabs Sound Effects API |
| Database | Supabase (PostgreSQL) |
| File storage | Supabase Storage |
| Auth | Supabase Auth (stretch goal) |

---

## Modal Deployment Config

| Parameter | Value |
|-----------|-------|
| App name | `hy-worldplay-ar` |
| GPU | `A100-80GB:2` (sp=2 sequence parallelism) |
| Model type | AR distilled (`model_type="ar"`, `few_step=True`) |
| Inference steps | 4 |
| Chunk size | 13 frames (4 latent frames, `chunk_latent_frames=4`) |
| torch.compile | Enabled |
| SageAttention | Enabled (graceful fallback) |
| Offloading | Disabled (`None`) |
| `min_containers` | 1 (always warm) |
| `scaledown_window` | 900s (15 min) |
| `max_inputs` | 1 (sequential generation) |
| Container timeout | 3600s |

---

## Frame Extraction

To chain chunks, the backend extracts the last frame from each generated chunk:

```python
import imageio

def extract_last_frame(video_bytes: bytes) -> bytes:
    """Extract the last frame from an MP4 as PNG bytes."""
    reader = imageio.get_reader(io.BytesIO(video_bytes), format="mp4")
    last_frame = None
    for frame in reader:
        last_frame = frame
    reader.close()
    img = Image.fromarray(last_frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
```

This last frame PNG is stored in-memory for the next chunk (and optionally persisted to Supabase for session replay).
