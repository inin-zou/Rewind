"""
HY-WorldPlay Modal deployment - AR Distilled model with 2x A100-80GB (sp=2)

Uses a persistent torchrun daemon: workers load the model ONCE at startup,
then process requests via file-based IPC. No model reload per request.
"""

import modal

app = modal.App("hy-worldplay-ar")

# Image with dependencies
worldplay_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.56.0",
        "accelerate>=0.34.0",
        "diffusers==0.35.0",
        "huggingface_hub==0.34.0",
        "safetensors==0.4.5",
        "einops",
        "omegaconf",
        "pyyaml",
        "pillow",
        "opencv-python",
        "imageio",
        "imageio-ffmpeg",
        "scipy",
        "numpy",
        "tqdm",
        "fastapi[standard]",
    )
    .run_commands(
        "git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git /app/HY-WorldPlay",
        "cd /app/HY-WorldPlay && pip install -r requirements.txt || true",
        # Install SageAttention from PyPI (pre-built wheel)
        "pip install triton sageattention || true",
    )
)

model_volume = modal.Volume.from_name("worldplay-models", create_if_missing=True)
MODEL_DIR = "/models"

# IPC paths
WORK_DIR = "/tmp/worker"
TRIGGER_FILE = f"{WORK_DIR}/trigger"
INPUT_FILE = f"{WORK_DIR}/input.json"
OUTPUT_FILE = f"{WORK_DIR}/output.json"
DONE_FILE = f"{WORK_DIR}/done"
READY_FILE = f"{WORK_DIR}/ready"

# Persistent worker script - loads model ONCE, then loops processing requests
WORKER_SCRIPT = r'''#!/usr/bin/env python3
"""
Persistent worker for torchrun with sp=2.
Loads the pipeline ONCE, then enters a loop processing requests via file IPC.
"""
import argparse
import base64
import json
import os
import struct
import sys
import time
import traceback
import json as _json

sys.path.insert(0, "/app/HY-WorldPlay")

WORK_DIR = "/tmp/worker"
TRIGGER_FILE = os.path.join(WORK_DIR, "trigger")
INPUT_FILE = os.path.join(WORK_DIR, "input.json")
OUTPUT_FILE = os.path.join(WORK_DIR, "output.json")
DONE_FILE = os.path.join(WORK_DIR, "done")
READY_FILE = os.path.join(WORK_DIR, "ready")

def patch_safetensors():
    import safetensors.torch
    import torch
    _orig = safetensors.torch.load_file

    def _patched(filename, device="cpu"):
        if os.path.isdir(filename):
            for f in os.listdir(filename):
                if f.endswith('.safetensors'):
                    filename = os.path.join(filename, f)
                    break
        try:
            return _orig(filename, device="cpu")
        except OSError as e:
            if "No such device" in str(e) or "os error 19" in str(e):
                print(f"mmap failed for {filename}, using sequential read...")
                tensors = {}
                with open(filename, "rb") as f:
                    header_size = struct.unpack('<Q', f.read(8))[0]
                    header_json = f.read(header_size).decode('utf-8')
                    header = _json.loads(header_json)
                    dtype_map = {
                        "F32": torch.float32, "F16": torch.float16,
                        "BF16": torch.bfloat16, "I64": torch.int64,
                        "I32": torch.int32, "I8": torch.int8, "U8": torch.uint8,
                    }
                    for key, info in header.items():
                        if key == "__metadata__":
                            continue
                        dtype = dtype_map.get(info["dtype"], torch.float32)
                        f.seek(8 + header_size + info["data_offsets"][0])
                        data = f.read(info["data_offsets"][1] - info["data_offsets"][0])
                        tensors[key] = torch.frombuffer(bytearray(data), dtype=dtype).reshape(info["shape"]).clone()
                return tensors
            raise

    safetensors.torch.load_file = _patched

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--action_ckpt", required=True)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"[Rank {rank}] Starting persistent worker on GPU {local_rank}")

    patch_safetensors()
    torch.cuda.set_device(local_rank)

    # Initialize parallel state with sp=2
    from hyvideo.commons.parallel_states import initialize_parallel_state
    from hyvideo.commons.infer_state import initialize_infer_state
    import argparse as ap

    print(f"[Rank {rank}] Initializing parallel state (sp=2)...")
    initialize_parallel_state(sp=2)

    # Check if SageAttention is available
    try:
        from sageattention import sageattn  # noqa: F401
        use_sage = True
        print(f"[Rank {rank}] SageAttention available")
    except ImportError:
        use_sage = False
        print(f"[Rank {rank}] SageAttention not available, using default attention")

    infer_args = ap.Namespace(
        sage_blocks_range="0-53",
        use_sageattn=use_sage,
        include_patterns="double_blocks",
        enable_torch_compile=True,
        use_fp8_gemm=False,
        quant_type="none",
        use_vae_parallel=False,
    )
    initialize_infer_state(infer_args)

    # Load pipeline ONCE
    print(f"[Rank {rank}] Loading AR distilled pipeline...")
    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        args.model_path,
        "480p_i2v",
        False,            # create_sr_pipeline
        False,            # force_sparse_attn
        torch.bfloat16,   # transformer_dtype
        None,             # enable_offloading (disabled - 2x A100-80GB has enough VRAM)
        None,             # enable_group_offloading
        True,             # overlap_group_offloading
        "cuda",           # device
        args.action_ckpt, # action_ckpt
    )
    print(f"[Rank {rank}] Pipeline loaded!")

    # Signal ready
    os.makedirs(WORK_DIR, exist_ok=True)
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
        num_frames = params["num_frames"]
        pose = params["pose"]
        num_inference_steps = params["num_inference_steps"]

        print(f"[Rank {rank}] Processing request: frames={num_frames}, pose={pose}, steps={num_inference_steps}")

        # Sync before generation
        dist.barrier()

        try:
            # Decode image
            import tempfile
            from PIL import Image
            img_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img_tmp.write(base64.b64decode(image_base64))
            img_tmp.close()
            input_image = Image.open(img_tmp.name).convert("RGB")

            # Parse pose - auto-adjust duration to match num_frames
            from hyvideo.generate import pose_to_input
            latent_num = (num_frames - 1) // 4 + 1

            parts = [p.strip() for p in pose.split(",")]
            if len(parts) == 1:
                cmd_parts = parts[0].split("-")
                if len(cmd_parts) == 2:
                    direction = cmd_parts[0]
                    pose = f"{direction}-{latent_num - 1}"
                    print(f"[Rank {rank}] Adjusted pose to '{pose}' for {latent_num} latents")

            viewmats, Ks, action = pose_to_input(pose, latent_num)

            start_time = time.time()

            output = pipe(
                prompt=prompt,
                reference_image=input_image,
                aspect_ratio="16:9",
                video_length=num_frames,
                num_inference_steps=num_inference_steps,
                seed=42,
                viewmats=viewmats.unsqueeze(0),
                Ks=Ks.unsqueeze(0),
                action=action.unsqueeze(0),
                few_step=True,
                chunk_latent_frames=4,
                model_type="ar",
                enable_sr=False,
            )

            gen_time = time.time() - start_time
            print(f"[Rank {rank}] Generation completed in {gen_time:.1f}s")

            # Only rank 0 writes output
            if rank == 0:
                import einops
                import imageio

                video = output.videos
                if video.ndim == 5:
                    assert video.shape[0] == 1
                    video = video[0]
                vid = (video * 255).clamp(0, 255).to(torch.uint8)
                vid = einops.rearrange(vid, "c f h w -> f h w c")

                out_dir = tempfile.mkdtemp()
                out_path = os.path.join(out_dir, "output.mp4")
                imageio.mimwrite(out_path, vid.cpu().numpy(), fps=24)

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

            # Clean up temp image
            os.unlink(img_tmp.name)

        except Exception as e:
            print(f"[Rank {rank}] Error: {traceback.format_exc()}")
            if rank == 0:
                with open(OUTPUT_FILE, "w") as f:
                    json.dump({"error": str(e), "traceback": traceback.format_exc()}, f)

        # Sync after generation
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
    image=worldplay_image,
    gpu="A100-80GB:2",
    timeout=3600,
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=900,
    min_containers=1,
)
@modal.concurrent(max_inputs=1)
class WorldPlayAR:
    @modal.enter()
    def setup(self):
        """Download models, write worker script, launch persistent torchrun daemon."""
        import os
        import subprocess
        import time

        from huggingface_hub import snapshot_download, hf_hub_download
        import shutil as _shutil

        # --- Download models (same as before) ---
        worldplay_path = os.path.join(MODEL_DIR, "HY-WorldPlay")
        ar_distill_dir = os.path.join(worldplay_path, "ar_distilled_action_model")
        ar_ckpt_dst = os.path.join(ar_distill_dir, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(ar_ckpt_dst):
            print("Downloading HY-WorldPlay AR distilled action model...")
            os.makedirs(ar_distill_dir, exist_ok=True)
            hf_hub_download(
                repo_id="tencent/HY-WorldPlay",
                filename="ar_distilled_action_model/model.safetensors",
                local_dir=worldplay_path,
                local_dir_use_symlinks=False,
                token=os.environ.get("HF_TOKEN"),
            )
            model_src = os.path.join(ar_distill_dir, "model.safetensors")
            if os.path.exists(model_src):
                _shutil.copy2(model_src, ar_ckpt_dst)
                print(f"Copied model.safetensors -> diffusion_pytorch_model.safetensors")
            if not os.path.exists(os.path.join(worldplay_path, "bidirectional_model")):
                snapshot_download(
                    repo_id="tencent/HY-WorldPlay",
                    local_dir=worldplay_path,
                    local_dir_use_symlinks=False,
                    token=os.environ.get("HF_TOKEN"),
                )
            model_volume.commit()

        hunyuan_path = os.path.join(MODEL_DIR, "HunyuanVideo-1.5")

        # 480p I2V transformer
        i2v_weights = os.path.join(hunyuan_path, "transformer", "480p_i2v", "diffusion_pytorch_model.safetensors")
        if not os.path.exists(i2v_weights):
            print("Downloading 480p_i2v transformer (~33GB)...")
            for fname in ["diffusion_pytorch_model.safetensors", "config.json"]:
                hf_hub_download(
                    repo_id="tencent/HunyuanVideo-1.5",
                    filename=f"transformer/480p_i2v/{fname}",
                    local_dir=hunyuan_path,
                    local_dir_use_symlinks=False,
                    token=os.environ.get("HF_TOKEN"),
                )
            model_volume.commit()

        # VAE
        vae_weights = os.path.join(hunyuan_path, "vae", "diffusion_pytorch_model.safetensors")
        if not os.path.exists(vae_weights):
            print("Downloading VAE (~5GB)...")
            for fname in ["diffusion_pytorch_model.safetensors", "config.json"]:
                hf_hub_download(
                    repo_id="tencent/HunyuanVideo-1.5",
                    filename=f"vae/{fname}",
                    local_dir=hunyuan_path,
                    local_dir_use_symlinks=False,
                    token=os.environ.get("HF_TOKEN"),
                )
            model_volume.commit()

        # Scheduler
        scheduler_config = os.path.join(hunyuan_path, "scheduler", "scheduler_config.json")
        if not os.path.exists(scheduler_config):
            hf_hub_download(
                repo_id="tencent/HunyuanVideo-1.5",
                filename="scheduler/scheduler_config.json",
                local_dir=hunyuan_path,
                local_dir_use_symlinks=False,
                token=os.environ.get("HF_TOKEN"),
            )
            model_volume.commit()

        # LLM text encoder
        llm_path = os.path.join(hunyuan_path, "text_encoder", "llm")
        if not os.path.exists(os.path.join(llm_path, "config.json")):
            print("Downloading Qwen2.5-VL-7B-Instruct (~16GB)...")
            snapshot_download(
                repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
                local_dir=llm_path,
                token=os.environ.get("HF_TOKEN"),
            )
            model_volume.commit()

        # ByT5
        byt5_path = os.path.join(hunyuan_path, "text_encoder", "byt5-small")
        if not os.path.exists(os.path.join(byt5_path, "config.json")):
            print("Downloading google/byt5-small...")
            snapshot_download(
                repo_id="google/byt5-small",
                local_dir=byt5_path,
                token=os.environ.get("HF_TOKEN"),
            )
            model_volume.commit()

        # SigLIP vision encoder
        siglip_path = os.path.join(hunyuan_path, "vision_encoder", "siglip")
        siglip_encoder = os.path.join(siglip_path, "image_encoder", "config.json")
        if not os.path.exists(siglip_encoder):
            print("Downloading SigLIP vision encoder...")
            if os.path.exists(siglip_path):
                _shutil.rmtree(siglip_path)
            os.makedirs(siglip_path, exist_ok=True)
            temp_siglip = os.path.join(siglip_path, "_temp")
            snapshot_download(
                repo_id="google/siglip-so400m-patch14-384",
                local_dir=temp_siglip,
                token=os.environ.get("HF_TOKEN"),
            )
            os.makedirs(os.path.join(siglip_path, "image_encoder"), exist_ok=True)
            os.makedirs(os.path.join(siglip_path, "feature_extractor"), exist_ok=True)
            for f in ["config.json", "model.safetensors"]:
                src = os.path.join(temp_siglip, f)
                if os.path.exists(src):
                    _shutil.copy2(src, os.path.join(siglip_path, "image_encoder", f))
            if os.path.exists(os.path.join(temp_siglip, "preprocessor_config.json")):
                _shutil.copy2(
                    os.path.join(temp_siglip, "preprocessor_config.json"),
                    os.path.join(siglip_path, "feature_extractor", "preprocessor_config.json")
                )
            _shutil.rmtree(temp_siglip)
            model_volume.commit()

        self.model_path = hunyuan_path
        self.action_ckpt = ar_ckpt_dst

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

        print("Launching persistent torchrun daemon (sp=2, 2 GPUs)...")
        self.stdout_log = open(f"{WORK_DIR}/stdout.log", "w")
        self.stderr_log = open(f"{WORK_DIR}/stderr.log", "w")
        self.worker_proc = subprocess.Popen(
            [
                "torchrun",
                "--nproc_per_node=2",
                "--master_port=29500",
                self.worker_script,
                "--model_path", self.model_path,
                "--action_ckpt", self.action_ckpt,
            ],
            stdout=self.stdout_log,
            stderr=self.stderr_log,
        )

        # Wait for workers to signal ready (model loaded)
        print("Waiting for workers to load model...")
        timeout = 600
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
                raise RuntimeError("Workers failed to start within 10 minutes")
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                print(f"  Still loading... ({int(elapsed)}s elapsed)")

        print(f"Workers ready! Model loaded in {time.time() - start:.0f}s")

    @modal.method()
    def generate(
        self,
        prompt: str,
        image_base64: str,
        num_frames: int = 125,
        pose: str = "w-31",
        num_inference_steps: int = 4,
    ) -> dict:
        """Generate video using persistent workers (no model reload)."""
        import json
        import os
        import time

        if not image_base64:
            return {"error": "image_base64 is required for I2V generation"}

        # Validate AR frame constraint
        latent_num = (num_frames - 1) // 4 + 1
        if latent_num % 4 != 0:
            return {"error": f"num_frames={num_frames} gives latent_num={latent_num} which is not divisible by 4. Try 125, 61, or 29."}

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
                "num_inference_steps": num_inference_steps,
            }, f)

        # Signal workers
        start_time = time.time()
        open(TRIGGER_FILE, "w").close()
        print(f"Request dispatched to workers: frames={num_frames}, steps={num_inference_steps}")

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

        # Signal workers we consumed the result (they wait for DONE to disappear)
        if os.path.exists(DONE_FILE):
            os.remove(DONE_FILE)

        print(f"Request completed in {total_time:.1f}s")
        return result

    @modal.method()
    def health(self) -> dict:
        """Health check."""
        import torch
        worker_alive = self.worker_proc.poll() is None if hasattr(self, 'worker_proc') else False
        return {
            "status": "healthy" if worker_alive else "workers_dead",
            "model_path": self.model_path,
            "model_type": "ar_distilled",
            "num_gpus": torch.cuda.device_count(),
            "action_ckpt": self.action_ckpt,
            "workers_alive": worker_alive,
            "pipeline_loaded": worker_alive,
        }


@app.function(
    image=worldplay_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
@modal.fastapi_endpoint(method="POST")
def generate_video_api(request: dict):
    """POST /generate_video_api"""
    model = WorldPlayAR()
    return model.generate.remote(
        prompt=request.get("prompt", "A scene"),
        image_base64=request.get("image_base64", ""),
        num_frames=request.get("num_frames", 125),
        pose=request.get("pose", "w-31"),
        num_inference_steps=request.get("num_inference_steps", 4),
    )


@app.function(image=worldplay_image, timeout=3600)
@modal.fastapi_endpoint(method="GET")
def health():
    """GET /health"""
    model = WorldPlayAR()
    return model.health.remote()


@app.local_entrypoint()
def main():
    model = WorldPlayAR()
    print(model.health.remote())
