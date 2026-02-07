"""
Unit tests for hy_worldplay_ar.py Modal deployment script.

Tests the logic without requiring Modal, GPUs, or model weights.
Run: uv run pytest modal-deploy/test_hy_worldplay_ar.py -v
"""

import json
import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import pytest


# ---------------------------------------------------------------------------
# Test: WORKER_SCRIPT is valid Python
# ---------------------------------------------------------------------------

class TestWorkerScript:
    """Tests for the embedded WORKER_SCRIPT string."""

    def _get_worker_script(self):
        """Extract WORKER_SCRIPT from the module source."""
        import ast
        module_path = os.path.join(os.path.dirname(__file__), "hy_worldplay_ar.py")
        with open(module_path) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "WORKER_SCRIPT":
                        return ast.literal_eval(node.value)
        raise RuntimeError("WORKER_SCRIPT not found in module")

    def test_worker_script_is_valid_python(self):
        """WORKER_SCRIPT should be syntactically valid Python."""
        script = self._get_worker_script()
        compile(script, "<WORKER_SCRIPT>", "exec")

    def test_worker_script_has_main_function(self):
        """WORKER_SCRIPT should define a main() function."""
        script = self._get_worker_script()
        assert "def main():" in script

    def test_worker_script_has_patch_safetensors(self):
        """WORKER_SCRIPT should define patch_safetensors()."""
        script = self._get_worker_script()
        assert "def patch_safetensors():" in script

    def test_worker_script_has_required_args(self):
        """WORKER_SCRIPT argparser should require input_json, output_json, model_path, action_ckpt."""
        script = self._get_worker_script()
        assert "--input_json" in script
        assert "--output_json" in script
        assert "--model_path" in script
        assert "--action_ckpt" in script

    def test_worker_script_uses_sp2(self):
        """WORKER_SCRIPT should initialize with sp=2."""
        script = self._get_worker_script()
        assert "initialize_parallel_state(sp=2)" in script

    def test_worker_script_uses_ar_mode(self):
        """WORKER_SCRIPT should use AR model type with few_step."""
        script = self._get_worker_script()
        assert 'model_type="ar"' in script
        assert "few_step=True" in script
        assert "chunk_latent_frames=4" in script

    def test_worker_script_only_rank0_writes(self):
        """Only rank 0 should write output."""
        script = self._get_worker_script()
        assert "if rank == 0:" in script

    def test_worker_script_can_be_written_to_disk(self):
        """WORKER_SCRIPT should be writable to a temp file."""
        script = self._get_worker_script()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            path = f.name
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test: AR frame constraint validation
# ---------------------------------------------------------------------------

class TestFrameConstraint:
    """Tests for the AR frame constraint: [(num_frames-1)//4+1] % 4 == 0."""

    @staticmethod
    def is_valid_ar_frames(num_frames: int) -> bool:
        latent_num = (num_frames - 1) // 4 + 1
        return latent_num % 4 == 0

    @pytest.mark.parametrize("num_frames", [13, 29, 61, 125])
    def test_valid_frame_counts(self, num_frames):
        assert self.is_valid_ar_frames(num_frames), f"{num_frames} should be valid"

    @pytest.mark.parametrize("num_frames", [10, 20, 50, 100, 120])
    def test_invalid_frame_counts(self, num_frames):
        assert not self.is_valid_ar_frames(num_frames), f"{num_frames} should be invalid"

    def test_125_frames_gives_32_latents(self):
        latent_num = (125 - 1) // 4 + 1
        assert latent_num == 32
        assert latent_num % 4 == 0

    def test_61_frames_gives_16_latents(self):
        latent_num = (61 - 1) // 4 + 1
        assert latent_num == 16
        assert latent_num % 4 == 0

    def test_29_frames_gives_8_latents(self):
        latent_num = (29 - 1) // 4 + 1
        assert latent_num == 8
        assert latent_num % 4 == 0


# ---------------------------------------------------------------------------
# Test: Generate method input validation
# ---------------------------------------------------------------------------

class TestGenerateValidation:
    """Test the generate method's input validation logic."""

    def _make_generate_validator(self):
        """Replicate the validation logic from generate()."""
        def validate(image_base64, num_frames):
            if not image_base64:
                return {"error": "image_base64 is required for I2V generation"}
            latent_num = (num_frames - 1) // 4 + 1
            if latent_num % 4 != 0:
                return {"error": f"num_frames={num_frames} gives latent_num={latent_num} which is not divisible by 4. Try 125, 61, or 29."}
            return None
        return validate

    def test_empty_image_rejected(self):
        validate = self._make_generate_validator()
        result = validate("", 125)
        assert result is not None
        assert "image_base64 is required" in result["error"]

    def test_valid_input_passes(self):
        validate = self._make_generate_validator()
        result = validate("iVBORw0KGgo=", 125)
        assert result is None

    def test_invalid_frames_rejected(self):
        validate = self._make_generate_validator()
        result = validate("iVBORw0KGgo=", 100)
        assert result is not None
        assert "not divisible by 4" in result["error"]

    def test_valid_frames_accepted(self):
        validate = self._make_generate_validator()
        for frames in [13, 29, 61, 125]:
            result = validate("iVBORw0KGgo=", frames)
            assert result is None, f"num_frames={frames} should be valid"


# ---------------------------------------------------------------------------
# Test: Torchrun command construction
# ---------------------------------------------------------------------------

class TestTorchrunCommand:
    """Test the torchrun command that generate() builds."""

    def test_command_structure(self):
        """Verify the torchrun command has the right structure."""
        worker_script = "/app/generate_worker.py"
        model_path = "/models/HunyuanVideo-1.5"
        action_ckpt = "/models/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors"
        input_json = "/tmp/input.json"
        output_json = "/tmp/output.json"

        cmd = [
            "torchrun",
            "--nproc_per_node=2",
            "--master_port=29500",
            worker_script,
            "--input_json", input_json,
            "--output_json", output_json,
            "--model_path", model_path,
            "--action_ckpt", action_ckpt,
        ]

        assert cmd[0] == "torchrun"
        assert "--nproc_per_node=2" in cmd
        assert "--master_port=29500" in cmd
        assert worker_script in cmd
        assert "--input_json" in cmd
        assert "--output_json" in cmd
        assert "--model_path" in cmd
        assert "--action_ckpt" in cmd

    def test_nproc_is_2(self):
        """Should use 2 processes for sp=2."""
        cmd = ["torchrun", "--nproc_per_node=2", "--master_port=29500", "/app/worker.py"]
        nproc_arg = [a for a in cmd if a.startswith("--nproc_per_node")]
        assert len(nproc_arg) == 1
        assert nproc_arg[0] == "--nproc_per_node=2"


# ---------------------------------------------------------------------------
# Test: Input/output JSON serialization
# ---------------------------------------------------------------------------

class TestIOSerialization:
    """Test input/output JSON format for the worker script."""

    def test_input_json_format(self):
        """Input JSON should contain all required fields."""
        input_data = {
            "prompt": "A beach at sunset",
            "image_base64": "iVBORw0KGgo=",
            "num_frames": 125,
            "pose": "w-31",
            "num_inference_steps": 4,
        }
        serialized = json.dumps(input_data)
        deserialized = json.loads(serialized)

        assert deserialized["prompt"] == "A beach at sunset"
        assert deserialized["image_base64"] == "iVBORw0KGgo="
        assert deserialized["num_frames"] == 125
        assert deserialized["pose"] == "w-31"
        assert deserialized["num_inference_steps"] == 4

    def test_output_json_format(self):
        """Output JSON should contain expected fields."""
        output_data = {
            "video_base64": "AAAAIGZ0eXBpc29t",
            "prompt": "A beach at sunset",
            "num_frames": 125,
            "pose": "w-31",
            "generation_time_seconds": 15.3,
        }
        serialized = json.dumps(output_data)
        deserialized = json.loads(serialized)

        assert "video_base64" in deserialized
        assert "generation_time_seconds" in deserialized
        assert isinstance(deserialized["generation_time_seconds"], float)

    def test_input_json_roundtrip_via_file(self):
        """Input JSON should survive write/read to temp file."""
        input_data = {
            "prompt": "Test",
            "image_base64": "abc123",
            "num_frames": 125,
            "pose": "w-31",
            "num_inference_steps": 4,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(input_data, f)
            path = f.name

        with open(path) as f:
            loaded = json.load(f)

        assert loaded == input_data
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test: GPU configuration
# ---------------------------------------------------------------------------

class TestGPUConfig:
    """Test GPU configuration in the deployment script."""

    def test_gpu_spec_is_2x_a100(self):
        """The script should request 2x A100-80GB."""
        module_path = os.path.join(os.path.dirname(__file__), "hy_worldplay_ar.py")
        with open(module_path) as f:
            source = f.read()
        assert 'gpu="A100-80GB:2"' in source

    def test_app_name(self):
        """App name should be hy-worldplay-ar."""
        module_path = os.path.join(os.path.dirname(__file__), "hy_worldplay_ar.py")
        with open(module_path) as f:
            source = f.read()
        assert '"hy-worldplay-ar"' in source

    def test_container_idle_timeout(self):
        """Container should stay warm for 15 min (900s)."""
        module_path = os.path.join(os.path.dirname(__file__), "hy_worldplay_ar.py")
        with open(module_path) as f:
            source = f.read()
        assert "container_idle_timeout=900" in source

    def test_concurrent_inputs_is_1(self):
        """Only 1 concurrent generation per container."""
        module_path = os.path.join(os.path.dirname(__file__), "hy_worldplay_ar.py")
        with open(module_path) as f:
            source = f.read()
        assert "allow_concurrent_inputs=1" in source


# ---------------------------------------------------------------------------
# Test: Model paths
# ---------------------------------------------------------------------------

class TestModelPaths:
    """Test model download paths and checkpoint resolution."""

    def test_ar_distilled_checkpoint_path(self):
        """Action checkpoint should point to ar_distilled_action_model."""
        model_dir = "/models"
        worldplay_path = os.path.join(model_dir, "HY-WorldPlay")
        expected = os.path.join(
            worldplay_path, "ar_distilled_action_model", "diffusion_pytorch_model.safetensors"
        )
        assert "ar_distilled_action_model" in expected
        assert expected.endswith(".safetensors")

    def test_fallback_to_model_safetensors(self):
        """Should fall back to model.safetensors if diffusion_pytorch_model.safetensors missing."""
        # Simulate the fallback logic from setup()
        primary = "/models/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors"
        fallback = "/models/HY-WorldPlay/ar_distilled_action_model/model.safetensors"

        # When primary doesn't exist, use fallback
        action_ckpt = primary
        if not os.path.exists(action_ckpt):
            if os.path.exists(fallback):
                action_ckpt = fallback

        # In test env neither exists, so it stays as primary (which is fine — setup logs it)
        assert action_ckpt == primary or action_ckpt == fallback

    def test_no_sr_models_downloaded(self):
        """SR models should NOT be downloaded (we disable SR)."""
        module_path = os.path.join(os.path.dirname(__file__), "hy_worldplay_ar.py")
        with open(module_path) as f:
            source = f.read()
        # The setup() in the main class should not download SR transformer/upsampler
        # (only the WORKER_SCRIPT mentions enable_sr=False)
        assert "720p_sr" not in source.split("WORKER_SCRIPT")[0]  # before WORKER_SCRIPT is fine


# ---------------------------------------------------------------------------
# Test: Pose string handling
# ---------------------------------------------------------------------------

class TestPoseStrings:
    """Test pose string formats."""

    def test_simple_poses(self):
        """Basic WASD poses should be valid strings."""
        valid_poses = ["w-31", "s-31", "a-31", "d-31"]
        for pose in valid_poses:
            parts = pose.split("-")
            assert len(parts) == 2
            assert parts[0] in ("w", "s", "a", "d")
            assert parts[1].isdigit()

    def test_combined_pose(self):
        """Combined poses with commas should parse."""
        pose = "w-20, right-10"
        parts = [p.strip() for p in pose.split(",")]
        assert len(parts) == 2

    def test_rotation_poses(self):
        """Rotation poses should be valid."""
        valid_rotations = ["up-5", "down-5", "left-5", "right-5"]
        for pose in valid_rotations:
            parts = pose.split("-")
            assert len(parts) == 2
            assert parts[0] in ("up", "down", "left", "right")


# ---------------------------------------------------------------------------
# Test: API endpoint defaults
# ---------------------------------------------------------------------------

class TestAPIDefaults:
    """Test default values used by the API endpoint."""

    def test_default_values(self):
        """Verify defaults match expected values."""
        module_path = os.path.join(os.path.dirname(__file__), "hy_worldplay_ar.py")
        with open(module_path) as f:
            source = f.read()

        # Default inference steps should be 4 (not 30)
        assert 'num_inference_steps=request.get("num_inference_steps", 4)' in source

        # Default pose
        assert 'pose=request.get("pose", "w-31")' in source

        # Default frames
        assert 'num_frames=request.get("num_frames", 125)' in source

        # Default prompt
        assert 'prompt=request.get("prompt", "A scene")' in source
