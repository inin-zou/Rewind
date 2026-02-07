# HY-WorldPlay AR Distilled — Modal Deployment

## Overview

This is the **fast** version of our WorldPlay deployment, using the autoregressive (AR) distilled model with only **4 inference steps** instead of 30. This should be dramatically faster than the bidirectional deployment (`hy_worldplay_simple.py`).

## Key Differences from Bidirectional Deployment

| Parameter | Bidirectional (`hy_worldplay_simple.py`) | AR Distilled (`hy_worldplay_ar.py`) |
|-----------|------------------------------------------|--------------------------------------|
| App name | `hy-worldplay-simple` | `hy-worldplay-ar` |
| Action checkpoint | `bidirectional_model/` | `ar_distilled_action_model/diffusion_pytorch_model.safetensors` |
| Model type | (default, bidirectional) | `ar` (autoregressive) |
| Inference steps | 30 | 4 |
| `few_step` | not set | `True` (forces `guidance_scale=1.0`) |
| `chunk_latent_frames` | not set (default 16) | `4` (AR processes in smaller chunks) |
| SR models downloaded | Yes (~34GB) | No (skipped, not needed) |
| Expected gen time | ~2.5 minutes / 125 frames | TBD (estimated ~20-40s) |

## Endpoint URLs

After `modal deploy modal-deploy/hy_worldplay_ar.py`:

```
POST https://ykzou1214--hy-worldplay-ar-generate-video-api.modal.run
GET  https://ykzou1214--hy-worldplay-ar-health.modal.run
```

## API

Same request/response format as the bidirectional endpoint.

### Request

```json
{
  "prompt": "A peaceful beach at sunset",
  "image_base64": "<base64 PNG/JPG>",
  "num_frames": 125,
  "pose": "w-31",
  "num_inference_steps": 4
}
```

### Response

```json
{
  "video_base64": "<base64 MP4>",
  "prompt": "A peaceful beach at sunset",
  "num_frames": 125,
  "pose": "w-31",
  "generation_time_seconds": 25.3
}
```

The response now includes `generation_time_seconds` for benchmarking.

### Health Check

```json
{
  "status": "healthy",
  "model_path": "/models/HunyuanVideo-1.5",
  "model_type": "ar_distilled",
  "action_ckpt": "/models/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors",
  "pipeline_loaded": true
}
```

## Frame Constraint

AR mode requires: `[(num_frames - 1) // 4 + 1] % 4 == 0`

Valid frame counts:

| Frames | Duration (24fps) | Latents |
|--------|-------------------|---------|
| 13 | ~0.5s | 4 |
| 29 | ~1.2s | 8 |
| 61 | ~2.5s | 16 |
| 125 | ~5.2s | 32 |

The endpoint validates this and returns an error for invalid frame counts.

## How AR Distilled Works

1. **Autoregressive generation** — generates video chunk-by-chunk (4 latent frames per chunk) instead of all at once
2. **Distilled model** — trained via Context Forcing to match the bidirectional teacher's quality in only 4 diffusion steps
3. **`few_step=True`** — automatically sets `guidance_scale=1.0` (no classifier-free guidance needed for distilled models)
4. **`chunk_latent_frames=4`** — each AR chunk processes 4 latent frames (vs 16 for bidirectional)
5. **`model_type="ar"`** — routes to `ar_rollout()` which uses KV-cache and causal attention

## Deployment

```bash
# Deploy AR distilled endpoint (coexists with bidirectional)
modal deploy modal-deploy/hy_worldplay_ar.py

# Test health
curl https://ykzou1214--hy-worldplay-ar-health.modal.run

# Test locally
modal run modal-deploy/hy_worldplay_ar.py
```

## Dependencies

Updated from the bidirectional deployment to match HY-WorldPlay's latest requirements:

- `torch==2.6.0` (was 2.4.0)
- `diffusers==0.35.0` (was >=0.30.0)
- `transformers==4.56.0` (was >=4.45.0)
- `safetensors==0.4.5`
- `huggingface-hub==0.34.0`
