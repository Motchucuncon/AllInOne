"""Image generation module using Flux.1-schnell via Diffusers.

Generates B-roll keyframe images locally from text prompts using the
Flux.1-schnell model loaded with the Hugging Face Diffusers library.
"""

import gc
import os
import torch
from diffusers import FluxPipeline
from PIL import Image

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEFAULT_OUTPUT_DIR = "output/images"
DEFAULT_IMAGE_SIZE = (1024, 576)  # 16:9 aspect ratio for video
DEFAULT_NUM_INFERENCE_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 0.0  # Flux.1-schnell uses guidance_scale=0.0

# Global pipeline cache (loaded once per process)
_pipeline: FluxPipeline | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _load_pipeline(model_id: str = DEFAULT_MODEL_ID) -> FluxPipeline:
    """Load the Flux.1-schnell pipeline, caching it globally.

    Uses bfloat16 and automatic device mapping to fit on Colab T4 (16 GB VRAM).
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print(f"[image_gen] Loading Flux.1-schnell from {model_id} ...")

    _pipeline = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    _pipeline.enable_model_cpu_offload()  # offload to CPU when not in use
    # Enable memory-efficient attention if available
    if hasattr(_pipeline, "enable_attention_slicing"):
        _pipeline.enable_attention_slicing()

    print("[image_gen] Pipeline loaded successfully.")
    return _pipeline


def _unload_pipeline():
    """Unload the pipeline and free GPU memory."""
    global _pipeline
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
    gc.collect()
    torch.cuda.empty_cache()
    print("[image_gen] Pipeline unloaded, VRAM cleared.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_broll_images(
    prompts: list[str],
    scene_ids: list[int],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    unload_after: bool = True,
) -> list[str]:
    """Generate B-roll keyframe images from a list of text prompts.

    Parameters
    ----------
    prompts : list of str
        One prompt per scene.
    scene_ids : list of int
        Corresponding scene IDs for file naming.
    output_dir : str
        Directory to save the generated PNG images.
    num_inference_steps : int
        Number of denoising steps (Flux.1-schnell works well with 4).
    guidance_scale : float
        Classifier-free guidance scale (0.0 for Flux.1-schnell).
    image_size : tuple of (width, height)
        Output image resolution.
    unload_after : bool
        If True, unload the pipeline from GPU after generation.

    Returns
    -------
    list of str
        Paths to the generated image files.
    """
    _ensure_dir(output_dir)

    if not prompts:
        print("[image_gen] No prompts provided, skipping generation.")
        return []

    # Load pipeline
    pipe = _load_pipeline()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        pipe = pipe.to("cuda")

    output_paths = []

    for i, (prompt, scene_id) in enumerate(zip(prompts, scene_ids)):
        print(f"[image_gen] Generating scene {scene_id} / {len(prompts)}: {prompt[:60]}...")

        try:
            # Generate the image
            result = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=image_size[0],
                height=image_size[1],
                generator=torch.Generator(device="cpu").manual_seed(42 + scene_id),
            )

            image: Image.Image = result.images[0]

            # Save
            filename = f"broll_scene_{scene_id:04d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath, "PNG")
            output_paths.append(filepath)
            print(f"[image_gen] Saved {filepath}")

        except Exception as exc:
            print(f"[image_gen] Failed to generate scene {scene_id}: {exc}")
            # Create a placeholder image (solid color) so the pipeline can continue
            placeholder = Image.new("RGB", image_size, color=(30, 30, 60))
            filename = f"broll_scene_{scene_id:04d}_fallback.png"
            filepath = os.path.join(output_dir, filename)
            placeholder.save(filepath, "PNG")
            output_paths.append(filepath)
            print(f"[image_gen] Saved fallback {filepath}")

        # Clean up after each image to prevent OOM
        gc.collect()
        torch.cuda.empty_cache()

    # Optionally unload pipeline
    if unload_after:
        _unload_pipeline()

    return output_paths


def generate_single_image(
    prompt: str,
    output_path: str = "output/images/single.png",
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    unload_after: bool = True,
) -> str:
    """Generate a single B-roll image from a single prompt.

    Returns
    -------
    str
        Path to the generated image file.
    """
    paths = generate_broll_images(
        [prompt],
        [1],
        output_dir=os.path.dirname(output_path),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        image_size=image_size,
        unload_after=unload_after,
    )
    return paths[0] if paths else output_path