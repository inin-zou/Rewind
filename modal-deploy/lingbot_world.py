"""
LingBot-World Modal deployment — 720p video generation with 4x A100-80GB

Uses a persistent torchrun daemon: workers load the dual DiT models ONCE at startup,
then process requests via file-based IPC. No model reload per request.

Based on Wan2.2 architecture with camera-controllable generation.
Uses 4 GPUs with FSDP + Ulysses sequence parallelism (ulysses_size=4).
40 heads % 4 = 0 ✓
"""

import modal

app = modal.App("lingbot-world")

# Pre-built flash-attn wheel — avoids CXX11 ABI mismatch during compilation
# flash-attn>=2.8 is incompatible with torch 2.6 (known issue #1783)
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)

lingbot_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.49.0,<=4.51.3",
        "accelerate>=1.1.1",
        "diffusers>=0.31.0",
        "huggingface_hub",
        "safetensors",
        "einops",
        "easydict",
        "ftfy",
        "pillow",
        "opencv-python",
        "imageio",
        "imageio-ffmpeg",
        "scipy",
        "numpy<2",
        "tqdm",
        "tokenizers>=0.20.3",
        "fastapi[standard]",
    )
    .run_commands(
        f"pip install {FLASH_ATTN_WHEEL}",
        "python -c 'import flash_attn; print(f\"flash-attn {flash_attn.__version__} OK\")'",
        "git clone https://github.com/Robbyant/lingbot-world.git /app/lingbot-world",
    )
)

model_volume = modal.Volume.from_name("lingbot-world-models", create_if_missing=True)
MODEL_DIR = "/models"

# IPC paths
WORK_DIR = "/tmp/worker"
TRIGGER_FILE = f"{WORK_DIR}/trigger"
INPUT_FILE = f"{WORK_DIR}/input.json"
OUTPUT_FILE = f"{WORK_DIR}/output.json"
DONE_FILE = f"{WORK_DIR}/done"
READY_FILE = f"{WORK_DIR}/ready"

# Persistent worker script — loads model ONCE, then loops processing requests
WORKER_SCRIPT = r'''#!/usr/bin/env python3
"""
Persistent worker for torchrun with sp=2.
Loads the WanI2V pipeline ONCE, then enters a loop processing requests via file IPC.
"""
import argparse
import base64
import json
import os
import sys
import tempfile
import time
import traceback

sys.path.insert(0, "/app/lingbot-world")

import numpy as np

WORK_DIR = "/tmp/worker"
TRIGGER_FILE = os.path.join(WORK_DIR, "trigger")
INPUT_FILE = os.path.join(WORK_DIR, "input.json")
OUTPUT_FILE = os.path.join(WORK_DIR, "output.json")
DONE_FILE = os.path.join(WORK_DIR, "done")
READY_FILE = os.path.join(WORK_DIR, "ready")


def wasd_to_poses(direction, num_frames=17):
    """Convert WASD direction string to camera pose matrices.

    Returns (poses [N,4,4], intrinsics [N,4]) numpy arrays.
    OpenCV coordinate system: X=right, Y=down, Z=forward.
    """
    poses = np.zeros((num_frames, 4, 4), dtype=np.float32)
    step = 0.3  # translation magnitude per frame

    for i in range(num_frames):
        poses[i] = np.eye(4)
        t = i * step

        if direction == "w":
            poses[i][2, 3] = t       # forward (+Z)
        elif direction == "s":
            poses[i][2, 3] = -t      # backward (-Z)
        elif direction == "a":
            poses[i][0, 3] = -t      # left (-X)
        elif direction == "d":
            poses[i][0, 3] = t       # right (+X)

    # Default intrinsics calibrated for 480p — will be rescaled by get_Ks_transformed
    intrinsics = np.tile(
        np.array([502.91, 503.11, 415.78, 239.78], dtype=np.float32),
        (num_frames, 1),
    )
    return poses, intrinsics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"[Rank {rank}] Starting persistent worker on GPU {local_rank} (world_size={world_size})")
    torch.cuda.set_device(local_rank)

    # Initialize distributed — must match ulysses_size == world_size
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
        from wan.distributed.util import init_distributed_group
        init_distributed_group()

    # Load pipeline ONCE with FSDP + Ulysses sequence parallelism
    print(f"[Rank {rank}] Loading WanI2V pipeline (FSDP={world_size > 1}, SP={world_size > 1})...")
    from wan import WanI2V
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS["i2v-A14B"]

    # Validate ulysses constraint: num_heads (40) must be divisible by world_size
    assert cfg.num_heads % world_size == 0, \
        f"num_heads ({cfg.num_heads}) must be divisible by world_size ({world_size})"

    wan_i2v = WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=(world_size > 1),
        dit_fsdp=(world_size > 1),
        use_sp=(world_size > 1),
        t5_cpu=True,
    )
    print(f"[Rank {rank}] Pipeline loaded!")

    # Signal ready
    os.makedirs(WORK_DIR, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        open(READY_FILE, "w").close()
    print(f"[Rank {rank}] Entering request loop...")

    # ---- Request loop ----
    while True:
        # Poll for trigger
        while not os.path.exists(TRIGGER_FILE):
            time.sleep(0.05)

        time.sleep(0.05)  # ensure input.json is fully written

        # All ranks read input
        with open(INPUT_FILE) as f:
            params = json.load(f)

        prompt = params["prompt"]
        image_base64 = params["image_base64"]
        num_frames = params.get("num_frames", 17)
        pose = params.get("pose", "w")
        sampling_steps = params.get("sampling_steps", 40)
        max_area_str = params.get("max_area", "720*1280")

        # Parse max_area
        if isinstance(max_area_str, str) and "*" in max_area_str:
            parts = max_area_str.split("*")
            max_area = int(parts[0]) * int(parts[1])
        elif isinstance(max_area_str, (int, float)):
            max_area = int(max_area_str)
        else:
            max_area = 720 * 1280

        # Determine shift based on resolution
        shift = 10.0 if max_area >= 720 * 1280 else 3.0

        print(f"[Rank {rank}] Processing: frames={num_frames}, pose={pose}, steps={sampling_steps}, max_area={max_area}")

        # Sync before generation
        if dist.is_initialized():
            dist.barrier()

        try:
            from PIL import Image
            import imageio

            # Decode image
            img_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img_tmp.write(base64.b64decode(image_base64))
            img_tmp.close()
            pil_image = Image.open(img_tmp.name).convert("RGB")

            # Create camera poses from WASD direction
            action_dir = None
            direction = pose.split("-")[0] if "-" in pose else pose
            if direction in ("w", "a", "s", "d"):
                action_dir = tempfile.mkdtemp()
                cam_poses, cam_intrinsics = wasd_to_poses(direction, num_frames)
                np.save(os.path.join(action_dir, "poses.npy"), cam_poses)
                np.save(os.path.join(action_dir, "intrinsics.npy"), cam_intrinsics)

            start_time = time.time()

            # offload_model must be False for multi-GPU FSDP setups
            # (FSDP-sharded models cannot be moved to CPU)
            video = wan_i2v.generate(
                input_prompt=prompt,
                img=pil_image,
                action_path=action_dir,
                max_area=max_area,
                frame_num=num_frames,
                shift=shift,
                sample_solver="unipc",
                sampling_steps=sampling_steps,
                guide_scale=5.0,
                seed=42,
                offload_model=False,
            )

            gen_time = time.time() - start_time
            print(f"[Rank {rank}] Generation completed in {gen_time:.1f}s")

            # Only rank 0 writes output
            if rank == 0:
                # video shape: (C, F, H, W) in range [-1, 1]
                vid = ((video + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
                vid = vid.permute(1, 2, 3, 0).cpu().numpy()  # (F, H, W, C)

                out_dir = tempfile.mkdtemp()
                out_path = os.path.join(out_dir, "output.mp4")
                imageio.mimwrite(out_path, vid, fps=16)

                with open(out_path, "rb") as f:
                    video_base64 = base64.b64encode(f.read()).decode()

                result = {
                    "video_base64": video_base64,
                    "prompt": prompt,
                    "num_frames": num_frames,
                    "pose": pose,
                    "generation_time_seconds": round(gen_time, 1),
                }
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(result, f)

                # Clean up temp files
                os.unlink(out_path)

            os.unlink(img_tmp.name)
            if action_dir:
                import shutil
                shutil.rmtree(action_dir, ignore_errors=True)

        except Exception as e:
            print(f"[Rank {rank}] Error: {traceback.format_exc()}")
            if rank == 0:
                with open(OUTPUT_FILE, "w") as f:
                    json.dump({"error": str(e), "traceback": traceback.format_exc()}, f)

        # Sync after generation
        if dist.is_initialized():
            dist.barrier()

        # Rank 0: signal done and remove trigger
        if rank == 0:
            open(DONE_FILE, "w").close()
            if os.path.exists(TRIGGER_FILE):
                os.remove(TRIGGER_FILE)

        # All ranks wait for done to be consumed before next iteration
        while os.path.exists(DONE_FILE):
            time.sleep(0.05)

        print(f"[Rank {rank}] Ready for next request")


if __name__ == "__main__":
    main()
'''


@app.cls(
    image=lingbot_image,
    gpu="A100-80GB:4",
    timeout=3600,
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=900,
    min_containers=1,
)
@modal.concurrent(max_inputs=1)
class LingBotWorld:
    @modal.enter()
    def setup(self):
        """Download models, write worker script, launch persistent torchrun daemon."""
        import os
        import subprocess
        import time

        from huggingface_hub import snapshot_download

        # --- Download model weights ---
        ckpt_dir = os.path.join(MODEL_DIR, "lingbot-world-base-cam")

        # Check if model already downloaded
        t5_path = os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        if not os.path.exists(t5_path):
            print("Downloading lingbot-world-base-cam model (~70GB)...")
            snapshot_download(
                repo_id="robbyant/lingbot-world-base-cam",
                local_dir=ckpt_dir,
                local_dir_use_symlinks=False,
                token=os.environ.get("HF_TOKEN"),
            )
            model_volume.commit()
            print("Model download complete!")
        else:
            print("Model already cached.")

        # Also need the umt5-xxl tokenizer (auto-downloaded by transformers, but let's cache it)
        tokenizer_dir = os.path.join(ckpt_dir, "google/umt5-xxl")
        if not os.path.exists(tokenizer_dir):
            print("Downloading umt5-xxl tokenizer...")
            snapshot_download(
                repo_id="google/umt5-xxl",
                local_dir=tokenizer_dir,
                local_dir_use_symlinks=False,
                token=os.environ.get("HF_TOKEN"),
                allow_patterns=["tokenizer*", "spiece*", "special_tokens*"],
            )
            model_volume.commit()

        self.ckpt_dir = ckpt_dir

        # --- Write worker script ---
        self.worker_script = "/app/generate_worker.py"
        with open(self.worker_script, "w") as f:
            f.write(WORKER_SCRIPT)

        # --- Launch persistent torchrun daemon ---
        os.makedirs(WORK_DIR, exist_ok=True)
        # Clean up stale IPC files
        for fp in [TRIGGER_FILE, INPUT_FILE, OUTPUT_FILE, DONE_FILE, READY_FILE]:
            if os.path.exists(fp):
                os.remove(fp)

        print("Launching persistent torchrun daemon (4 GPUs, ulysses_size=4)...")
        self.stdout_log = open(f"{WORK_DIR}/stdout.log", "w")
        self.stderr_log = open(f"{WORK_DIR}/stderr.log", "w")
        self.worker_proc = subprocess.Popen(
            [
                "torchrun",
                "--nproc_per_node=4",
                "--master_port=29500",
                self.worker_script,
                "--ckpt_dir", self.ckpt_dir,
            ],
            stdout=self.stdout_log,
            stderr=self.stderr_log,
        )

        # Wait for workers to signal ready (model loaded)
        print("Waiting for workers to load model...")
        timeout = 900  # 15 min for first load (large model)
        start = time.time()
        while not os.path.exists(READY_FILE):
            time.sleep(1)
            elapsed = time.time() - start
            if self.worker_proc.poll() is not None:
                self.stdout_log.flush()
                self.stderr_log.flush()
                with open(f"{WORK_DIR}/stdout.log") as f:
                    stdout = f.read()
                with open(f"{WORK_DIR}/stderr.log") as f:
                    stderr = f.read()
                raise RuntimeError(
                    f"Worker process died during startup.\nSTDOUT: {stdout[-3000:]}\nSTDERR: {stderr[-3000:]}"
                )
            if elapsed > timeout:
                self.worker_proc.kill()
                raise RuntimeError("Workers failed to start within 15 minutes")
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                print(f"  Still loading... ({int(elapsed)}s elapsed)")

        print(f"Workers ready! Model loaded in {time.time() - start:.0f}s")

    @modal.method()
    def generate(
        self,
        prompt: str,
        image_base64: str,
        num_frames: int = 17,
        pose: str = "w",
        sampling_steps: int = 40,
        max_area: str = "720*1280",
    ) -> dict:
        """Generate video using persistent workers (no model reload)."""
        import json
        import os
        import time

        if not image_base64:
            return {"error": "image_base64 is required"}

        # Validate frame constraint: must be 4n+1
        if (num_frames - 1) % 4 != 0:
            corrected = ((num_frames - 1) // 4) * 4 + 1
            return {"error": f"num_frames must be 4n+1. Got {num_frames}, try {corrected}."}

        # Check worker is alive
        if self.worker_proc.poll() is not None:
            return {"error": "Worker process is not running"}

        # Clean up stale files
        for fp in [DONE_FILE, OUTPUT_FILE]:
            if os.path.exists(fp):
                os.remove(fp)

        # Write input
        with open(INPUT_FILE, "w") as f:
            json.dump({
                "prompt": prompt,
                "image_base64": image_base64,
                "num_frames": num_frames,
                "pose": pose,
                "sampling_steps": sampling_steps,
                "max_area": max_area,
            }, f)

        # Signal workers
        start_time = time.time()
        open(TRIGGER_FILE, "w").close()
        print(f"Request dispatched: frames={num_frames}, pose={pose}, steps={sampling_steps}")

        # Wait for done
        timeout = 600
        while not os.path.exists(DONE_FILE):
            time.sleep(0.1)
            if time.time() - start_time > timeout:
                return {"error": "Generation timed out (600s)"}
            if self.worker_proc.poll() is not None:
                return {"error": "Worker process crashed during generation"}

        total_time = time.time() - start_time

        # Read output
        try:
            with open(OUTPUT_FILE) as f:
                result = json.load(f)
            result["total_time_seconds"] = round(total_time, 1)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            result = {"error": f"Failed to read output: {e}", "total_time_seconds": round(total_time, 1)}

        # Signal workers we consumed the result
        if os.path.exists(DONE_FILE):
            os.remove(DONE_FILE)

        print(f"Request completed in {total_time:.1f}s")
        return result

    @modal.method()
    def health(self) -> dict:
        """Health check."""
        import torch
        worker_alive = self.worker_proc.poll() is None if hasattr(self, "worker_proc") else False
        return {
            "status": "healthy" if worker_alive else "workers_dead",
            "model": "lingbot-world-base-cam",
            "model_type": "wan2.2_i2v_A14B",
            "num_gpus": torch.cuda.device_count(),
            "ckpt_dir": self.ckpt_dir,
            "workers_alive": worker_alive,
            "pipeline_loaded": worker_alive,
        }


@app.function(
    image=lingbot_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
@modal.fastapi_endpoint(method="POST")
def generate_video_api(request: dict):
    """POST /generate_video_api"""
    model = LingBotWorld()
    return model.generate.remote(
        prompt=request.get("prompt", "A scene"),
        image_base64=request.get("image_base64", ""),
        num_frames=request.get("num_frames", 17),
        pose=request.get("pose", "w"),
        sampling_steps=request.get("sampling_steps", 40),
        max_area=request.get("max_area", "720*1280"),
    )


@app.function(image=lingbot_image, timeout=3600)
@modal.fastapi_endpoint(method="GET")
def health():
    """GET /health"""
    model = LingBotWorld()
    return model.health.remote()


@app.local_entrypoint()
def main():
    model = LingBotWorld()
    print(model.health.remote())
