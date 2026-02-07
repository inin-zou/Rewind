# Rewind — Backend Architecture

## Overview

The backend orchestrates a **chunk-based world exploration** experience. A user uploads a photo, the system generates a short video chunk showing camera movement through the scene, and the user navigates with WASD + joystick to generate subsequent chunks. Each chunk's last frame seeds the next generation, creating an unlimited explorable experience.

---

## Constraints (WorldPlay)

| Parameter | Value |
|-----------|-------|
| Video chunk length | ~5.2s (125 frames @ 24fps) |
| Generation time (warm) | 2–5 minutes per chunk |
| Generation time (cold start) | +3–5 minutes first request |
| GPU | A100-80GB, 1 concurrent request per container |
| Output resolution | 480p (SR disabled) |

**Key implication:** This is **not real-time**. The experience is turn-based — the user watches a chunk, chooses a direction, waits for the next chunk.

---

## User Flow

```
1. User uploads a photo
2. Backend runs scene analysis (Claude) + first video chunk (Modal) + soundscape (ElevenLabs) in parallel
3. User watches the first 5s chunk with ambient audio
4. User controls direction via WASD (movement) + joystick (camera angle)
5. Backend extracts LAST FRAME of previous chunk as new input image
6. Backend generates next chunk with the user's chosen pose
7. Repeat 4–6 indefinitely
```

---

## Experience Duration

Technically **unlimited** — each chunk's last frame becomes the next chunk's input. But with 2–5 min generation per 5s of video, the pacing is slow. Strategies to improve:

| Strategy | How | Tradeoff |
|----------|-----|----------|
| **Pre-generate branches** | While user watches chunk N, generate 2–4 possible next directions in parallel | Higher GPU cost (2–4x) |
| **Shorter chunks** | Fewer frames (e.g. 61 frames = ~2.5s) → faster generation | Choppier experience |
| **Buffer ahead** | Always stay 1–2 chunks ahead of the user | Requires predicting user direction |
| **Fewer inference steps** | Reduce `num_inference_steps` from 30 to 15–20 | Lower visual quality |

---

## API Design

### `POST /api/memories` — Start a new memory

Creates a new memory session from an uploaded photo. Runs scene analysis, generates the first video chunk, and produces the soundscape.

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
    "video_base64": "<base64 MP4>",
    "pose": "w-31",
    "chunk_index": 0
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

### `POST /api/memories/{memory_id}/chunks` — Generate next chunk

Takes the user's chosen direction and generates the next video chunk. The backend automatically extracts the last frame from the previous chunk as the input image.

**Request:**
```json
{
  "pose": "a-15, right-2",
  "parent_chunk_id": "uuid"
}
```

**Response:**
```json
{
  "chunk": {
    "chunk_id": "uuid",
    "video_base64": "<base64 MP4>",
    "pose": "a-15, right-2",
    "chunk_index": 3
  }
}
```

### `GET /api/memories/{memory_id}` — Get memory details

Retrieve a memory and its full chunk tree for replaying a session.

**Response:**
```json
{
  "memory_id": "uuid",
  "original_image_url": "https://...",
  "scene": { ... },
  "chunks": [
    { "chunk_id": "uuid", "parent_chunk_id": null, "pose": "w-31", "chunk_index": 0, "video_url": "https://..." },
    { "chunk_id": "uuid", "parent_chunk_id": "uuid", "pose": "a-15", "chunk_index": 1, "video_url": "https://..." }
  ]
}
```

### `GET /api/health` — Health check

```json
{
  "status": "healthy",
  "modal": { "status": "healthy", "pipeline_loaded": true },
  "supabase": "connected"
}
```

---

## Pose Mapping (Frontend → Backend)

The frontend sends WASD keys + joystick angle. The backend converts these into a WorldPlay pose string.

| Frontend Input | Pose String | Description |
|----------------|-------------|-------------|
| W key | `w-31` | Walk forward |
| S key | `s-31` | Walk backward |
| A key | `a-31` | Strafe left |
| D key | `d-31` | Strafe right |
| Joystick up | `up-N` | Pitch camera up |
| Joystick down | `down-N` | Pitch camera down |
| Joystick left | `left-N` | Yaw camera left |
| Joystick right | `right-N` | Yaw camera right |
| W + joystick right | `w-20, right-10` | Walk forward while turning right |

The step count (N) maps to how far the camera moves across the chunk's latent frames. Default is 31 for a full chunk of movement. Combined inputs produce comma-separated pose strings.

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
| `pose` | text | Pose string used (e.g. `"w-31"`) |
| `video_path` | text | Path in Supabase Storage |
| `last_frame_path` | text | Path in Supabase Storage (input for next chunk) |
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
| `videos` | Generated MP4 chunks |
| `frames` | Extracted last frames (PNG) for seeding next chunk |
| `audio` | Generated soundscape files |

---

## Backend Internal Flow

### First Chunk (Memory Creation)

```
POST /api/memories { image_base64, prompt }
    │
    ├─── [parallel] Claude: analyze scene → SceneInfo
    ├─── [parallel] Modal: generate video (pose="w-31") → video_base64
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
    └─── Return response
```

### Subsequent Chunks

```
POST /api/memories/{id}/chunks { pose, parent_chunk_id }
    │
    ├─── Fetch parent chunk's last_frame from Supabase Storage
    ├─── Modal: generate video (image=last_frame, pose=user_pose) → video_base64
    │
    ├─── Extract last frame from new video
    ├─── Upload to Supabase Storage: video, last frame
    ├─── Insert into Supabase: new chunks row
    │
    └─── Return response
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| API server | FastAPI (Python) |
| Package manager | uv |
| Video generation | HY-WorldPlay on Modal (A100-80GB) |
| Scene analysis | Claude (Anthropic API) |
| Soundscape | ElevenLabs Sound Effects API |
| Database | Supabase (PostgreSQL) |
| File storage | Supabase Storage |
| Auth | Supabase Auth (stretch goal) |

---

## Frame Extraction

To chain chunks, the backend must extract the last frame from each generated video. This is done server-side:

```python
import imageio

def extract_last_frame(video_bytes: bytes) -> bytes:
    """Extract the last frame from an MP4 as PNG bytes."""
    reader = imageio.get_reader(io.BytesIO(video_bytes), format="mp4")
    last_frame = None
    for frame in reader:
        last_frame = frame
    reader.close()
    # Encode as PNG
    img = Image.fromarray(last_frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
```

This last frame PNG is stored in Supabase and used as `image_base64` for the next chunk's Modal call.
