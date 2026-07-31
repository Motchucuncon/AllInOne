"""Video generation module — multi-model engine.

Supports two backends:
- Wan 2.1 (image-to-video, via diffusers)
- LTX-Video (image-to-video, via diffusers)

Each clip is rendered from a keyframe image, then GPU memory is aggressively
freed with gc.collect() and torch.cuda.empty_cache().
"""

import gc
import os
import subprocess
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "output/clips"
DEFAULT_CLIP_DURATION = 5.0  # seconds per clip
DEFAULT_FPS = 16
DEFAULT_VIDEO_SIZE = (1024, 576)

# ---------------------------------------------------------------------------
# Wan 2.1 configuration
# ---------------------------------------------------------------------------
WAN_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

# ---------------------------------------------------------------------------
# LTX-Video configuration
# ---------------------------------------------------------------------------
LTX_MODEL_ID = "Lightricks/LTX-Video"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _cleanup_vram():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[video_gen] VRAM cleared (gc.collect + torch.cuda.empty_cache).")


def _resize_image_for_video(image_path: str, target_size: tuple[int, int]) -> str:
    """Resize a keyframe image to the target video resolution using FFmpeg.

    Returns the path to the resized image (or the original if no resize needed).
    """
    from PIL import Image

    img = Image.open(image_path)
    if img.size == target_size:
        return image_path

    resized = img.resize(target_size, Image.LANCZOS)
    base, ext = os.path.splitext(image_path)
    resized_path = f"{base}_resized{ext}"
    resized.save(resized_path, "PNG")
    return resized_path


def _make_fallback_clip(
    image_path: str,
    output_path: str,
    duration: float = DEFAULT_CLIP_DURATION,
    fps: int = DEFAULT_FPS,
    video_size: tuple[int, int] = DEFAULT_VIDEO_SIZE,
) -> str:
    """Create a simple motion video clip from a static image using FFmpeg.

    Applies a gentle zoompan effect to simulate camera motion.
    This serves as a fallback when the AI video model fails.
    """
    _ensure_dir(os.path.dirname(output_path))

    # Resize image first
    img = _resize_image_for_video(image_path, video_size)

    # Use FFmpeg zoompan to create motion
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", img,
        "-c:v", "libx264",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-vf", (
            f"scale={video_size[0]}:{video_size[1]}:force_original_aspect_ratio=decrease,"
            f"pad={video_size[0]}:{video_size[1]}:(ow-iw)/2:(oh-ih)/2,"
            f"zoompan=z='min(zoom+0.0015,1.5)':d={int(fps * duration)}:fps={fps}:s={video_size[0]}x{video_size[1]}"
        ),
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[video_gen] Fallback clip created: {output_path}")
    except subprocess.CalledProcessError as exc:
        print(f"[video_gen] Fallback FFmpeg failed: {exc.stderr}")
        # Ultimate fallback: simple copy frame
        cmd_simple = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", img,
            "-c:v", "libx264",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        subprocess.run(cmd_simple, check=True, capture_output=True, text=True)
        print(f"[video_gen] Simple fallback clip created: {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Wan 2.1 backend
# ---------------------------------------------------------------------------

def _render_with_wan(
    image_path: str,
    output_path: str,
    prompt: str = "",
    num_frames: int = 81,
    fps: int = DEFAULT_FPS,
) -> str:
    """Render a video clip using Wan 2.1 image-to-video.

    Parameters
    ----------
    image_path : str
        Path to the keyframe PNG image.
    output_path : str
        Where to save the output MP4 video.
    prompt : str
        Text prompt describing the desired motion.
    num_frames : int
        Number of frames to generate (Wan 2.1 default: 81).
    fps : int
        Output video frame rate.

    Returns
    -------
    str
        Path to the generated video clip.
    """
    _ensure_dir(os.path.dirname(output_path))

    try:
        from diffusers import WanI2VPipeline
        import torch
        from PIL import Image

        print(f"[video_gen] Loading Wan 2.1 pipeline from {WAN_MODEL_ID} ...")

        pipe = WanI2VPipeline.from_pretrained(
            WAN_MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()

        # Load and resize image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((1024, 576), Image.LANCZOS)

        print(f"[video_gen] Generating {num_frames} frames with Wan 2.1 ...")

        # Generate frames
        frames = pipe(
            image=image,
            prompt=prompt or "gentle camera movement, cinematic",
            num_frames=num_frames,
            generator=torch.Generator(device="cpu").manual_seed(42),
        ).frames[0]

        # Save as video using FFmpeg
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(tmp_dir, f"frame_{i:06d}.png")
            frame.save(frame_path, "PNG")
            frame_paths.append(frame_path)

        # Use FFmpeg to assemble frames into video
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Cleanup temp frames
        for fp in frame_paths:
            try:
                os.remove(fp)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

        print(f"[video_gen] Wan 2.1 clip saved: {output_path}")

        # Cleanup pipeline
        del pipe
        _cleanup_vram()

        return output_path

    except Exception as exc:
        print(f"[video_gen] Wan 2.1 generation failed: {exc}")
        _cleanup_vram()
        # Fallback to static zoompan
        return _make_fallback_clip(image_path, output_path, duration=num_frames / fps, fps=fps)


# ---------------------------------------------------------------------------
# LTX-Video backend
# ---------------------------------------------------------------------------

def _render_with_ltx(
    image_path: str,
    output_path: str,
    prompt: str = "",
    num_frames: int = 49,
    fps: int = DEFAULT_FPS,
) -> str:
    """Render a video clip using LTX-Video (image-to-video).

    Parameters
    ----------
    image_path : str
        Path to the keyframe PNG image.
    output_path : str
        Where to save the output MP4 video.
    prompt : str
        Text prompt describing the desired motion.
    num_frames : int
        Number of frames to generate (LTX-Video default: 49).
    fps : int
        Output video frame rate.

    Returns
    -------
    str
        Path to the generated video clip.
    """
    _ensure_dir(os.path.dirname(output_path))

    try:
        from diffusers import LTXImageToVideoPipeline
        import torch
        from PIL import Image

        print(f"[video_gen] Loading LTX-Video pipeline from {LTX_MODEL_ID} ...")

        pipe = LTXImageToVideoPipeline.from_pretrained(
            LTX_MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()

        # Load and resize image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((1024, 576), Image.LANCZOS)

        print(f"[video_gen] Generating {num_frames} frames with LTX-Video ...")

        # Generate frames
        result = pipe(
            image=image,
            prompt=prompt or "gentle camera movement, cinematic",
            num_frames=num_frames,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )

        frames = result.frames[0]

        # Save as video using FFmpeg
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(tmp_dir, f"frame_{i:06d}.png")
            frame.save(frame_path, "PNG")
            frame_paths.append(frame_path)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Cleanup temp frames
        for fp in frame_paths:
            try:
                os.remove(fp)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

        print(f"[video_gen] LTX-Video clip saved: {output_path}")

        # Cleanup pipeline
        del pipe
        _cleanup_vram()

        return output_path

    except Exception as exc:
        print(f"[video_gen] LTX-Video generation failed: {exc}")
        _cleanup_vram()
        return _make_fallback_clip(image_path, output_path, duration=num_frames / fps, fps=fps)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_video_clips(
    image_paths: list[str],
    prompts: list[str],
    scene_ids: list[int],
    model: str = "wan_2_1",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    clip_duration: float = DEFAULT_CLIP_DURATION,
    fps: int = DEFAULT_FPS,
) -> list[str]:
    """Render video clips from keyframe images using the selected model.

    Parameters
    ----------
    image_paths : list of str
        Paths to keyframe PNG images (one per scene).
    prompts : list of str
        Text prompts describing desired motion (one per scene).
    scene_ids : list of int
        Scene IDs for file naming.
    model : str
        One of "wan_2_1" or "ltx_video".
    output_dir : str
        Directory to save the generated MP4 clips.
    clip_duration : float
        Duration of each clip in seconds (used for fallback).
    fps : int
        Output video frame rate.

    Returns
    -------
    list of str
        Paths to the generated video clip files.
    """
    _ensure_dir(output_dir)

    if not image_paths:
        print("[video_gen] No image paths provided, skipping.")
        return []

    model = model.lower().replace("-", "_")
    if model not in ("wan_2_1", "ltx_video"):
        print(f"[video_gen] Unknown model '{model}'. Falling back to wan_2_1.")
        model = "wan_2_1"

    # Determine number of frames based on model
    num_frames_map = {
        "wan_2_1": 81,
        "ltx_video": 49,
    }
    num_frames = num_frames_map.get(model, 49)

    output_paths = []

    for i, (img_path, prompt, scene_id) in enumerate(zip(image_paths, prompts, scene_ids)):
        clip_filename = f"clip_scene_{scene_id:04d}.mp4"
        clip_path = os.path.join(output_dir, clip_filename)

        print(f"[video_gen] Rendering scene {scene_id} / {len(image_paths)} with {model} ...")

        if model == "wan_2_1":
            result_path = _render_with_wan(
                img_path, clip_path,
                prompt=prompt,
                num_frames=num_frames,
                fps=fps,
            )
        else:
            result_path = _render_with_ltx(
                img_path, clip_path,
                prompt=prompt,
                num_frames=num_frames,
                fps=fps,
            )

        output_paths.append(result_path)

        # Aggressive cleanup after each clip
        _cleanup_vram()

    return output_paths