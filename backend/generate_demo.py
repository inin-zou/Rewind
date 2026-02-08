"""
Generate demo video by running the full pipeline:
  1. Send image to /api/generate with a sequence of WASD poses
  2. Extract last frame from each video chunk
  3. Stitch all chunks into one continuous ~20s video

Usage:
  python generate_demo.py [--image PATH] [--api URL] [--output PATH]
"""

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

# Movement sequence: W forward x3, D right x3, W forward x3
MOVE_SEQUENCE = [
    ("w-3", "forward"),
    ("w-3", "forward"),
    ("w-3", "forward"),
    ("d-3", "right"),
    ("d-3", "right"),
    ("d-3", "right"),
    ("w-3", "forward"),
    ("w-3", "forward"),
    ("w-3", "forward"),
]

DEFAULT_PROMPT = "A serene ancient shrine with cherry blossoms at golden sunset"


def extract_last_frame(video_bytes: bytes, tmp_dir: Path) -> bytes:
    """Extract the last frame from an mp4 video as JPEG bytes."""
    video_path = tmp_dir / "chunk.mp4"
    frame_path = tmp_dir / "last_frame.jpg"

    video_path.write_bytes(video_bytes)

    # Get total frames
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    total_frames = int(probe.stdout.strip())
    print(f"    Video has {total_frames} frames")

    # Extract last frame
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-sseof", "-0.1",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(frame_path),
        ],
        capture_output=True,
    )

    return frame_path.read_bytes()


def stitch_videos(chunk_paths: list[Path], output_path: str):
    """Concatenate video chunks into one continuous video using ffmpeg."""
    tmp_dir = chunk_paths[0].parent
    list_file = tmp_dir / "chunks.txt"

    # Write ffmpeg concat list
    with open(list_file, "w") as f:
        for p in chunk_paths:
            f.write(f"file '{p}'\n")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            output_path,
        ],
        check=True,
    )
    print(f"\nStitched {len(chunk_paths)} chunks -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Rewind demo video")
    parser.add_argument(
        "--image",
        default=str(
            Path(__file__).parent.parent
            / "frontend/src/assets/world-3.jpg"
        ),
        help="Path to input image",
    )
    parser.add_argument("--api", default="http://localhost:8000", help="Backend API URL")
    parser.add_argument("--output", default="demo_output.mp4", help="Output video path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Scene prompt")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    print(f"Loaded image: {image_path.name} ({len(image_b64)} chars base64)")

    # Use a chunks dir next to the output file
    output_stem = Path(args.output).stem
    chunks_dir = Path(args.output).parent / f"{output_stem}_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="rewind_demo_"))
    print(f"Temp dir: {tmp_dir}")
    print(f"Chunks dir: {chunks_dir}")

    chunk_paths: list[Path] = []
    current_image_b64 = image_b64
    skip_conversion = False  # First call needs 16:9 conversion

    total_start = time.time()

    for i, (pose, direction) in enumerate(MOVE_SEQUENCE):
        step = i + 1
        print(f"\n{'='*50}")
        print(f"Step {step}/{len(MOVE_SEQUENCE)}: {direction} ({pose})")
        print(f"{'='*50}")

        start = time.time()

        try:
            with httpx.Client(timeout=600) as client:
                resp = client.post(
                    f"{args.api}/api/generate",
                    json={
                        "image_base64": current_image_b64,
                        "prompt": args.prompt,
                        "pose": pose,
                        "skip_conversion": skip_conversion,
                    },
                )
                if resp.status_code != 200:
                    print(f"  ERROR {resp.status_code}: {resp.text[:500]}")
                    print("  Stopping early.")
                    break
                result = resp.json()
        except Exception as e:
            print(f"  ERROR: {e}")
            print("  Stopping early.")
            break

        elapsed = time.time() - start
        gen_time = result.get("generation_time_seconds", "?")
        print(f"  Generated in {elapsed:.1f}s (server: {gen_time}s)")

        # Save chunk video (permanent + tmp for stitching)
        video_bytes = base64.b64decode(result["video_base64"])
        chunk_path = tmp_dir / f"chunk_{step:02d}.mp4"
        chunk_path.write_bytes(video_bytes)
        chunk_paths.append(chunk_path)
        # Also save to permanent chunks dir
        perm_chunk = chunks_dir / f"chunk_{step:02d}.mp4"
        perm_chunk.write_bytes(video_bytes)
        print(f"  Saved chunk: {chunk_path.name} ({len(video_bytes)} bytes)")

        # Extract last frame for next iteration
        last_frame_bytes = extract_last_frame(video_bytes, tmp_dir)
        current_image_b64 = base64.b64encode(last_frame_bytes).decode()
        skip_conversion = True  # Subsequent frames are already 16:9
        print(f"  Extracted last frame ({len(last_frame_bytes)} bytes)")

    total_elapsed = time.time() - total_start

    if not chunk_paths:
        print("No chunks generated. Exiting.")
        sys.exit(1)

    # Stitch all chunks
    output_path = str(Path(args.output).resolve())
    stitch_videos(chunk_paths, output_path)

    # Print summary
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            output_path,
        ],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0

    print(f"\nDone!")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Chunks: {len(chunk_paths)}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
