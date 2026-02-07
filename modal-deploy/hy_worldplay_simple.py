"""
HY-WorldPlay Modal deployment - Optimized version with model pre-loading
"""

import modal

app = modal.App("hy-worldplay-simple")

# Image with dependencies
worldplay_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "diffusers>=0.30.0",
        "huggingface_hub",
        "safetensors",
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
    )
)

model_volume = modal.Volume.from_name("worldplay-models", create_if_missing=True)
MODEL_DIR = "/models"


# Safetensors patch as a module-level function
SAFETENSORS_PATCH = '''
import safetensors.torch
import torch
import struct
import json as _json
import os

_orig_load_file = safetensors.torch.load_file

def _patched_load_file(filename, device="cpu"):
    if os.path.isdir(filename):
        for f in os.listdir(filename):
            if f.endswith('.safetensors'):
                filename = os.path.join(filename, f)
                break
    try:
        return _orig_load_file(filename, device="cpu")
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

safetensors.torch.load_file = _patched_load_file
print("Safetensors patched for mmap fallback")
'''


@app.cls(
    image=worldplay_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    container_idle_timeout=900,  # Keep warm for 15 min
    allow_concurrent_inputs=1,
)
class WorldPlaySimple:
    @modal.enter()
    def setup(self):
        """Download models and pre-load pipeline."""
        import os
        import sys
        import struct
        import json as _json

        sys.path.insert(0, "/app/HY-WorldPlay")

        # Apply safetensors patch BEFORE any model imports
        import safetensors.torch
        import torch
        _orig_load_file = safetensors.torch.load_file

        def _patched_load_file(filename, device="cpu"):
            if os.path.isdir(filename):
                for f in os.listdir(filename):
                    if f.endswith('.safetensors'):
                        filename = os.path.join(filename, f)
                        break
            try:
                return _orig_load_file(filename, device="cpu")
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

        safetensors.torch.load_file = _patched_load_file
        print("Safetensors patched for mmap fallback")

        from huggingface_hub import snapshot_download, hf_hub_download

        # Download models (same as before)
        worldplay_path = os.path.join(MODEL_DIR, "HY-WorldPlay")
        if not os.path.exists(os.path.join(worldplay_path, "bidirectional_model", "config.json")):
            print("Downloading HY-WorldPlay action models...")
            snapshot_download(
                repo_id="tencent/HY-WorldPlay",
                local_dir=worldplay_path,
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
        import shutil
        siglip_path = os.path.join(hunyuan_path, "vision_encoder", "siglip")
        siglip_encoder = os.path.join(siglip_path, "image_encoder", "config.json")
        if not os.path.exists(siglip_encoder):
            print("Downloading SigLIP vision encoder...")
            if os.path.exists(siglip_path):
                shutil.rmtree(siglip_path)
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
                    shutil.copy2(src, os.path.join(siglip_path, "image_encoder", f))
            if os.path.exists(os.path.join(temp_siglip, "preprocessor_config.json")):
                shutil.copy2(
                    os.path.join(temp_siglip, "preprocessor_config.json"),
                    os.path.join(siglip_path, "feature_extractor", "preprocessor_config.json")
                )
            shutil.rmtree(temp_siglip)
            model_volume.commit()

        # SR transformer
        sr_weights = os.path.join(hunyuan_path, "transformer", "720p_sr_distilled", "diffusion_pytorch_model.safetensors")
        if not os.path.exists(sr_weights):
            print("Downloading SR transformer (~33GB)...")
            for fname in ["diffusion_pytorch_model.safetensors", "config.json"]:
                hf_hub_download(
                    repo_id="tencent/HunyuanVideo-1.5",
                    filename=f"transformer/720p_sr_distilled/{fname}",
                    local_dir=hunyuan_path,
                    local_dir_use_symlinks=False,
                    token=os.environ.get("HF_TOKEN"),
                )
            model_volume.commit()

        # SR upsampler
        sr_upsampler = os.path.join(hunyuan_path, "upsampler", "720p_sr_distilled", "diffusion_pytorch_model.safetensors")
        if not os.path.exists(sr_upsampler):
            print("Downloading SR upsampler...")
            for fname in ["diffusion_pytorch_model.safetensors", "config.json"]:
                hf_hub_download(
                    repo_id="tencent/HunyuanVideo-1.5",
                    filename=f"upsampler/720p_sr_distilled/{fname}",
                    local_dir=hunyuan_path,
                    local_dir_use_symlinks=False,
                    token=os.environ.get("HF_TOKEN"),
                )
            model_volume.commit()

        self.model_path = hunyuan_path
        self.worldplay_path = worldplay_path
        self.action_ckpt = os.path.join(worldplay_path, "bidirectional_model")

        # PRE-LOAD the pipeline to GPU memory
        print("Pre-loading HY-WorldPlay pipeline to GPU...")
        import torch
        torch.cuda.init()

        # Set up distributed environment for single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        try:
            # Initialize parallel and inference state (required by HY-WorldPlay)
            from hyvideo.commons.parallel_states import initialize_parallel_state
            from hyvideo.commons.infer_state import initialize_infer_state
            import argparse

            print(f"Initializing parallel state...")
            parallel_dims = initialize_parallel_state(sp=1)
            torch.cuda.set_device(0)

            # Create args for infer_state (use default values from generate.py)
            args = argparse.Namespace(
                sage_blocks_range="0-53",  # Default range
                use_sageattn=False,
                include_patterns="double_blocks",
                enable_torch_compile=False,
                use_fp8_gemm=False,
                quant_type="none",
                use_vae_parallel=False,
            )
            print(f"Initializing infer state...")
            initialize_infer_state(args)

            print(f"Importing HunyuanVideo_1_5_Pipeline...")
            from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline

            print(f"Creating pipeline with:")
            print(f"  pretrained_model_name_or_path={self.model_path}")
            print(f"  transformer_version=480p_i2v")
            print(f"  action_ckpt={self.action_ckpt}")

            # Pass arguments positionally to avoid any keyword arg issues
            # Disable SR pipeline to save memory, enable offloading for large models
            self.pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
                self.model_path,  # pretrained_model_name_or_path (positional)
                "480p_i2v",       # transformer_version (positional)
                False,            # create_sr_pipeline (disabled to save memory)
                False,            # force_sparse_attn
                torch.bfloat16,   # transformer_dtype
                "model",          # enable_offloading (move unused models to CPU)
                None,             # enable_group_offloading
                True,             # overlap_group_offloading
                "cuda",           # device
                self.action_ckpt, # action_ckpt
            )
            print("Pipeline pre-loaded successfully!")
            self.preload_error = None
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.preload_error = f"{type(e).__name__}: {e}"
            print(f"Pipeline pre-load failed: {self.preload_error}")
            print(f"Full traceback:\n{tb}")
            self.pipe = None

        print(f"Setup complete. Model path: {self.model_path}")

    @modal.method()
    def generate(
        self,
        prompt: str,
        image_base64: str,
        num_frames: int = 125,
        pose: str = "w-31",
        num_inference_steps: int = 30,
    ) -> dict:
        """Generate video using pre-loaded pipeline."""
        import os
        import sys
        import base64
        import tempfile
        from pathlib import Path

        sys.path.insert(0, "/app/HY-WorldPlay")

        if not image_base64:
            return {"error": "image_base64 is required for I2V generation"}

        # Save input image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(base64.b64decode(image_base64))
            image_path = f.name

        output_dir = tempfile.mkdtemp()

        if self.pipe is None:
            return {"error": "Pipeline not loaded. Check container logs."}

        try:
            from hyvideo.generate import pose_to_input
            from PIL import Image
            import torch

            # Load input image
            input_image = Image.open(image_path).convert("RGB")

            # Parse pose
            latent_num = (num_frames - 1) // 4 + 1
            viewmats, Ks, action = pose_to_input(pose, latent_num)

            print(f"Generating video: {prompt}, frames={num_frames}, pose={pose}")

            # Generate (use reference_image for I2V mode)
            # Add batch dimension with unsqueeze(0)
            output = self.pipe(
                prompt=prompt,
                reference_image=input_image,  # I2V mode
                aspect_ratio="16:9",
                video_length=num_frames,
                num_inference_steps=num_inference_steps,
                viewmats=viewmats.unsqueeze(0),
                Ks=Ks.unsqueeze(0),
                action=action.unsqueeze(0),
                enable_sr=False,  # Disable SR to save memory
            )

            # Save video using the same method as HY-WorldPlay generate.py
            import einops
            import imageio

            def save_video(video, path):
                if video.ndim == 5:
                    assert video.shape[0] == 1
                    video = video[0]
                vid = (video * 255).clamp(0, 255).to(torch.uint8)
                vid = einops.rearrange(vid, "c f h w -> f h w c")
                imageio.mimwrite(path, vid.cpu().numpy(), fps=24)

            output_path = os.path.join(output_dir, "output.mp4")
            save_video(output.videos, output_path)

            with open(output_path, "rb") as f:
                video_base64 = base64.b64encode(f.read()).decode()

            return {
                "video_base64": video_base64,
                "prompt": prompt,
                "num_frames": num_frames,
                "pose": pose,
            }

        except Exception as e:
            import traceback
            return {
                "error": f"{type(e).__name__}: {str(e)}",
                "traceback": traceback.format_exc()[:3000],
            }

    @modal.method()
    def health(self) -> dict:
        """Health check."""
        import os
        result = {
            "status": "healthy",
            "model_path": self.model_path,
            "pipeline_loaded": self.pipe is not None,
        }
        if hasattr(self, 'preload_error') and self.preload_error:
            result["preload_error"] = self.preload_error[:2000]
        return result


@app.function(
    image=worldplay_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.fastapi_endpoint(method="POST")
def generate_video_api(request: dict):
    """POST /generate_video_api - matches Go client URL pattern."""
    model = WorldPlaySimple()
    return model.generate.remote(
        prompt=request.get("prompt", "A scene"),
        image_base64=request.get("image_base64", ""),
        num_frames=request.get("num_frames", 125),
        pose=request.get("pose", "w-31"),
        num_inference_steps=request.get("num_inference_steps", 30),
    )


@app.function(image=worldplay_image)
@modal.fastapi_endpoint(method="GET")
def health():
    """GET /health"""
    model = WorldPlaySimple()
    return model.health.remote()


@app.local_entrypoint()
def main():
    model = WorldPlaySimple()
    print(model.health.remote())
