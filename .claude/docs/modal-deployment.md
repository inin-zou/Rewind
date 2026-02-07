# HY-1.5-WorldPlay Modal Deployment

## Overview

We deploy HY-1.5-WorldPlay (Tencent's camera-controllable video generation model) as a serverless GPU service on [Modal](https://modal.com). Modal handles container orchestration, GPU provisioning, and auto-scaling so we don't manage any infrastructure ourselves. The service exposes an HTTP API that our app calls to generate explorable 3D-world videos from a single photo.

## Architecture

```
[ Rewind App ] --POST--> https://ykzou1214--hy-worldplay-simple-generate-video-api.modal.run
                                              |
                                      [ Modal FastAPI Endpoint ]
                                              |
                                      [ WorldPlaySimple class ]
                                      (A100-80GB GPU container)
                                              |
                                      [ HY-WorldPlay Pipeline ]
                                              |
                                      returns JSON { video_base64, prompt, num_frames, pose }
```

The deployment consists of three layers:

1. **Container Image** — A pre-built Docker-like image with all dependencies (PyTorch, diffusers, HY-WorldPlay repo, ffmpeg, etc.)
2. **GPU Class (`WorldPlaySimple`)** — A stateful container class running on an A100-80GB that downloads models on first boot and keeps the pipeline warm in GPU memory
3. **HTTP Endpoints** — FastAPI endpoints that accept requests and delegate to the GPU class

---

## Endpoint URLs

After running `modal deploy hy_worldplay_simple.py`, Modal generates public URLs. Our workspace is `ykzou1214`, the app is `hy-worldplay-simple`, and the URL pattern is `https://<workspace>--<app-name>-<function-name>.modal.run`.

### Generate Video Endpoint

```
POST https://ykzou1214--hy-worldplay-simple-generate-video-api.modal.run
```

### Health Check Endpoint

```
GET https://ykzou1214--hy-worldplay-simple-health.modal.run
```

---

## How to Call the API

### Request Format

Send a `POST` request with a JSON body containing the following fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | `"A scene"` | Text description of the scene to generate |
| `image_base64` | string | Yes | — | Base64-encoded input image (PNG or JPG). This is the source photo that will be turned into a video. |
| `num_frames` | int | No | `125` | Number of video frames (~5.2 seconds at 24fps) |
| `pose` | string | No | `"w-31"` | Camera movement trajectory. Controls how the camera moves through the scene. |
| `num_inference_steps` | int | No | `30` | Diffusion denoising steps. More steps = higher quality but slower. |

### Pose Values

The `pose` parameter controls camera movement. Format is `<direction>-<steps>`:

- `w-31` — Walk forward 31 steps
- `s-31` — Walk backward 31 steps
- `a-31` — Strafe left 31 steps
- `d-31` — Strafe right 31 steps

### cURL Example

```bash
# 1. Base64-encode your input image
IMAGE_B64=$(base64 -i my_photo.png)

# 2. Send the POST request
curl -X POST \
  "https://ykzou1214--hy-worldplay-simple-generate-video-api.modal.run" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"A peaceful beach at sunset with gentle waves\",
    \"image_base64\": \"${IMAGE_B64}\",
    \"num_frames\": 125,
    \"pose\": \"w-31\",
    \"num_inference_steps\": 30
  }" \
  -o response.json

# 3. Extract the video from the response
# The response JSON contains a "video_base64" field with the MP4 data
python3 -c "
import json, base64
with open('response.json') as f:
    data = json.load(f)
if 'error' in data:
    print('Error:', data['error'])
else:
    with open('output.mp4', 'wb') as f:
        f.write(base64.b64decode(data['video_base64']))
    print('Saved output.mp4')
"
```

### Python Example

```python
import requests
import base64

# Encode the input image
with open("my_photo.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Call the Modal endpoint
response = requests.post(
    "https://ykzou1214--hy-worldplay-simple-generate-video-api.modal.run",
    json={
        "prompt": "A peaceful beach at sunset with gentle waves",
        "image_base64": image_b64,
        "num_frames": 125,
        "pose": "w-31",
        "num_inference_steps": 30,
    },
    timeout=600,  # generation can take several minutes
)

result = response.json()

if "error" in result:
    print(f"Error: {result['error']}")
else:
    # Decode and save the output video
    video_bytes = base64.b64decode(result["video_base64"])
    with open("output.mp4", "wb") as f:
        f.write(video_bytes)
    print(f"Saved output.mp4 ({len(video_bytes)} bytes)")
```

### Response Format

**Success:**
```json
{
  "video_base64": "<base64-encoded MP4 video>",
  "prompt": "A peaceful beach at sunset with gentle waves",
  "num_frames": 125,
  "pose": "w-31"
}
```

**Error:**
```json
{
  "error": "RuntimeError: CUDA out of memory...",
  "traceback": "Traceback (most recent call last):\n..."
}
```

### Health Check

```bash
curl "https://ykzou1214--hy-worldplay-simple-health.modal.run"
```

Response:
```json
{
  "status": "healthy",
  "model_path": "/models/HunyuanVideo-1.5",
  "pipeline_loaded": true
}
```

If the pipeline failed to load, the response includes a `preload_error` field with the error message.

---

## How the Endpoint Works Internally

When a POST request hits `generate_video_api`:

1. **Modal routes the request** to a FastAPI function running on a lightweight container (no GPU).
2. **The function calls `WorldPlaySimple().generate.remote(...)`** — this is a Modal RPC call that routes to the GPU container class.
3. **Modal finds or spins up a GPU container**:
   - If a warm container exists (idle < 15 min), the request goes there immediately.
   - If no warm container exists, Modal provisions a new A100-80GB, runs `setup()` (downloads models if needed, loads pipeline), then processes the request. This cold start takes ~2-5 minutes.
4. **The `generate` method runs on the GPU**:
   - Decodes the base64 image
   - Parses the camera pose into view matrices and camera intrinsics
   - Runs the HunyuanVideo diffusion pipeline with WorldPlay action conditioning
   - Encodes the output video tensor to MP4 at 24fps
   - Returns the MP4 as base64
5. **The response flows back** through the FastAPI endpoint to the caller.

```
Client POST  -->  FastAPI (no GPU)  -->  .generate.remote()  -->  GPU Container
                                                                       |
Client <--   <--  FastAPI           <--  return dict          <--  pipeline output
```

---

## Container Image

```python
worldplay_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers", "accelerate", "diffusers", "huggingface_hub", ...)
    .run_commands("git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git /app/HY-WorldPlay")
)
```

The image is built once and cached by Modal. It:
- Starts from a slim Debian base with Python 3.10
- Installs system libraries needed for video processing (`ffmpeg`, OpenGL/GLX)
- Installs PyTorch 2.4.0 with CUDA 12.4 support
- Installs ML dependencies (transformers, diffusers, accelerate, etc.)
- Clones the HY-WorldPlay source code into the container at `/app/HY-WorldPlay`

## Model Storage (Persistent Volume)

```python
model_volume = modal.Volume.from_name("worldplay-models", create_if_missing=True)
```

Models are stored on a **Modal Volume** (`worldplay-models`) mounted at `/models`. This is a persistent network filesystem that survives across container restarts, so models only need to be downloaded once. The total model weight is approximately **90 GB**:

| Model | Source | Size | Purpose |
|-------|--------|------|---------|
| HY-WorldPlay action models | `tencent/HY-WorldPlay` | ~2 GB | Camera-controllable action conditioning |
| 480p I2V transformer | `tencent/HunyuanVideo-1.5` | ~33 GB | Image-to-video diffusion backbone |
| 720p SR transformer | `tencent/HunyuanVideo-1.5` | ~33 GB | Super-resolution upscaling |
| 720p SR upsampler | `tencent/HunyuanVideo-1.5` | ~1 GB | SR pixel upsampler |
| VAE | `tencent/HunyuanVideo-1.5` | ~5 GB | Video encoder/decoder |
| Qwen2.5-VL-7B-Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` | ~16 GB | Text encoder (LLM) |
| ByT5-small | `google/byt5-small` | ~300 MB | Byte-level text encoder |
| SigLIP | `google/siglip-so400m-patch14-384` | ~800 MB | Vision encoder for image conditioning |

On first container boot, the `setup()` method checks for each model on the volume and downloads any that are missing from HuggingFace Hub. After download, `model_volume.commit()` persists the files.

## GPU Container Class

```python
@app.cls(
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    container_idle_timeout=900,
    allow_concurrent_inputs=1,
)
class WorldPlaySimple:
```

Key configuration:
- **`gpu="A100-80GB"`** — Requires an 80GB VRAM A100 to fit the full pipeline in memory
- **`timeout=3600`** — Each generation can run up to 60 minutes before timeout
- **`container_idle_timeout=900`** — Container stays warm for 15 minutes after the last request (avoids cold starts for subsequent calls)
- **`allow_concurrent_inputs=1`** — Only one generation at a time per container (GPU memory constraint)
- **`secrets`** — Injects the `HF_TOKEN` environment variable for HuggingFace downloads

## Startup / Model Pre-loading (`@modal.enter`)

The `setup()` method runs once when a container starts. It:

1. **Patches safetensors** — Applies a monkey-patch to `safetensors.torch.load_file` that adds a fallback for when memory-mapped file loading (`mmap`) fails on Modal's network filesystem. The fallback reads tensor data sequentially from disk instead.

2. **Downloads missing models** — Checks each model path on the persistent volume and downloads from HuggingFace if absent.

3. **Initializes distributed environment** — Sets up single-GPU environment variables (`WORLD_SIZE=1`, `RANK=0`, etc.) required by HY-WorldPlay's parallel state system.

4. **Loads the full pipeline into GPU memory** — Creates the `HunyuanVideo_1_5_Pipeline` with these settings:
   - `transformer_version="480p_i2v"` — Uses the 480p image-to-video transformer
   - `create_sr_pipeline=False` — Disables super-resolution to save VRAM
   - `enable_offloading="model"` — Moves unused sub-models to CPU when not in use
   - `transformer_dtype=torch.bfloat16` — Uses BF16 precision to reduce memory
   - `action_ckpt` — Loads WorldPlay's camera action conditioning weights

After startup, the pipeline remains in memory so generation requests don't pay model loading cost.

---

## Deployment Commands

```bash
# Deploy to Modal (creates the app and public endpoints)
modal deploy hy_worldplay_simple.py

# Run locally for testing (calls health check via local entrypoint)
modal run hy_worldplay_simple.py
```

After `modal deploy`, the CLI prints the live endpoint URLs. You can also find them in the Modal dashboard under the `hy-worldplay-simple` app.

## Secrets Setup

A Modal secret named `huggingface-secret` must be configured with your HuggingFace token:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxxxx
```

This token is needed to download gated models from HuggingFace Hub.

## Cost and Performance Notes

- **Cold start**: ~3-5 minutes on first boot (model download from volume + pipeline loading to GPU). Subsequent boots with cached volume are faster (~2-3 min for pipeline load only).
- **Warm generation**: ~2-5 minutes per video depending on frame count and inference steps.
- **GPU cost**: A100-80GB on Modal. The container stays warm for 15 minutes after each request to minimize cold starts.
- **Storage**: The `worldplay-models` volume stores ~90 GB of model weights persistently.
- **Request timeout**: Set your HTTP client timeout to at least 10 minutes — generation is slow, and cold starts add to the wait.
